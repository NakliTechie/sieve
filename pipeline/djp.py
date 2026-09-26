#!/usr/bin/env python3
"""djp: onboard a customer, then build, gate and release their own classifier, per question on djev, GLiNER, tiny or mmBERT.

  python3 pipeline/djp.py status [--json]                      # the whole picture: one line per customer
  python3 pipeline/djp.py init <customer>                      # create $DJP_HOME/<customer>/customer.json to edit
  python3 pipeline/djp.py import <customer> <file.csv|.jsonl> --state <col> --label <col>[,<col>...]
                                                               # their data, their column names -> data.jsonl (replaces)
  python3 pipeline/djp.py check <customer>                     # validate data + questions; no model calls
  .venv/bin/python pipeline/djp.py eval <customer> [--arms djev,gliner,tiny,mmbert]
                                                               # every candidate on the holdout; writes no release
  .venv/bin/python pipeline/djp.py release <customer> [--arms djev,gliner,tiny,mmbert] [--force]
                                                               # candidates -> per-question gate -> release -> CURRENT
  .venv/bin/python pipeline/djp.py review <customer> <unlabelled.csv> [--share 0.5] [--teacher release|gliner|djev@3]
                                                               # model answers + the rows a person should check -> CSV
  .venv/bin/python pipeline/djp.py label <customer> <file.csv|.jsonl> [--state state] [--label <cols>] [--reviewed]
                                                               # append new labelled rows, retrain, gate, release

Arms (backends.py): djev (zero-shot on the GPU service; + calibration, + the customer's rules and examples),
gliner (GLiNER2.5-Decide, zero-shot, local; + calibration), tfidf (keywords + logistic regression, CPU, seconds), tiny (Ettin-17M fine-tuned on the training rows, local;
only for questions with >= 20 labelled rows per option), mmbert (mmBERT-small, multilingual, the same way). The bar is plain djev (or plain gliner when djev is not in
--arms). Per question, a candidate replaces the bar only if its holdout accuracy >= the bar's and its log-loss is
lower; among those the highest accuracy wins (log-loss breaks ties). The winning candidate is what gets served (serve.py): tiny weights are
saved in the release. `release` is skipped (UNCHANGED) when the live release was built from the same data and arms.

Customer data lives in $DJP_HOME (default data/customers, gitignored). Layer 2 (TRAINING.md) would add trained djev
weights as another candidate behind the same gate.

Env: DJP_HOME, DJEV_URL (default http://localhost:8081, the `gcloud run services proxy`), DJEV_WORKERS (default 8).
Exit: 0 ok (RELEASED, UNCHANGED, VALID, ...) · 2 setup (customer missing, service unreachable, wrong python)
      · 7 gate refused (nothing released; CURRENT unchanged) · 8 data invalid (nothing written).
"""
import csv
import json
import math
import os
import shutil
import sys
import time
import urllib.error
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import djcore as P  # noqa: E402

URL = os.environ.get("DJEV_URL", "http://localhost:8081")
WORKERS = int(os.environ.get("DJEV_WORKERS", "8"))
IMAGE = "ghcr.io/taeold/djev-run@sha256:a352b97ab4ccf9c0d5caabbb7df16e0b40a801426cd8582c0e3fe81b2da16ac1"
MIN_HOLDOUT = 50  # held-out rows per question before a release's numbers mean much
VERDICTS = {2: "SETUP", 7: "GATE_REFUSED", 8: "DATA_INVALID"}
COMMANDS = ("status", "init", "import", "check", "review", "label", "release", "eval", "rollback")  # tools.json covers each
TODO = "TODO: say in one sentence what this question decides"


def die(code, msg):
    print(f"verdict={VERDICTS[code]} {msg}", file=sys.stderr)
    sys.exit(code)


# ---------- onboarding ----------

def cmd_init(cid):
    d = P.HOME / cid
    if (d / "customer.json").exists():
        die(2, f"{d}/customer.json already exists")
    P.write_atomic(d / "customer.json", json.dumps({
        "id": cid,
        "description": "Who this customer is and what the deployment decides for them.",
        "questions": {},
        "context": "Their rules, in plain words, as they would explain them to a new colleague.",
        "few_shot": 6}, indent=2) + "\n")
    print(f"verdict=CREATED {d}/customer.json  next: python3 pipeline/djp.py import {cid} <file> --state <col> --label <col>")


def read_table(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def cmd_import(cid, path, state_col, label_cols):
    d = P.customer_dir(cid) or die(2, f"no customer '{cid}'. Run: python3 pipeline/djp.py init {cid}")
    customer, _ = P.load_customer(cid)
    src = read_table(path)
    rows, skipped = [], 0
    for r in src:
        answers = r.get("answers") if isinstance(r.get("answers"), dict) else {c: r.get(c) for c in label_cols}
        state = r.get(state_col)
        answers = {q: str(a).strip() for q, a in answers.items() if q in label_cols and a not in (None, "")}
        if not state or len(answers) != len(label_cols):
            skipped += 1
            continue
        rows.append({"state": str(state).strip(), "answers": answers})
    if not rows:
        die(8, f"no usable rows in {path} (state column '{state_col}', labels {label_cols}); columns seen: {list(src[0]) if src else []}")
    qs = customer.setdefault("questions", {})
    for c in label_cols:  # a new question gets its options from the data; existing ones keep theirs (check flags strays)
        if c not in qs:
            qs[c] = {"type": "choice", "instructions": TODO, "criteria": sorted({r["answers"][c] for r in rows})}
    P.write_atomic(d / "customer.json", json.dumps(customer, indent=2) + "\n")
    P.write_atomic(d / "data.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    print(f"verdict=IMPORTED customer={cid} rows={len(rows)} skipped={skipped} questions={list(qs)} "
          f"next: edit {d}/customer.json (instructions, context), then python3 pipeline/djp.py check {cid}")


def validate(customer, rows):
    """-> (errors, warnings, summary). Errors block a release; warnings are printed."""
    errors, warnings = [], []
    qs = customer.get("questions") or {}
    if not qs:
        errors.append("no questions in customer.json")
    for q, spec in qs.items():
        crit = spec.get("criteria") or []
        if len(crit) < 2:
            errors.append(f"question '{q}' needs at least 2 options")
        if spec.get("instructions", TODO) == TODO:
            warnings.append(f"question '{q}' still has the placeholder instructions")
    if customer.get("context", "").startswith("Their rules, in plain words"):
        warnings.append("context is still the placeholder; the customer's rules in words help the profile variant")
    for i, r in enumerate(rows, 1):
        if not str(r.get("state", "")).strip():
            errors.append(f"row {i}: empty state")
        for q, spec in qs.items():
            a = r.get("answers", {}).get(q)
            if a is None:
                errors.append(f"row {i}: no answer for '{q}'")
            elif a not in spec.get("criteria", []):
                errors.append(f"row {i}: '{a}' is not an option of '{q}' {spec.get('criteria')}")
    dup = [s for s, n in Counter(r.get("state") for r in rows).items() if n > 1]
    if dup:
        warnings.append(f"{len(dup)} duplicate states (they land in the same split)")
    train, hold, test = P.split(rows) if rows else ([], [], [])
    if len(hold) < MIN_HOLDOUT:
        warnings.append(f"only {len(hold)} held-out rows; {MIN_HOLDOUT}+ per question before trusting a release "
                        f"(about {math.ceil(MIN_HOLDOUT / P.HOLDOUT_TENTHS * 10)} rows in total)")
    counts = {q: dict(Counter(r["answers"].get(q) for r in rows)) for q in qs}
    for q, c in counts.items():
        for opt in qs[q].get("criteria", []):
            if c.get(opt, 0) < 3:
                warnings.append(f"'{q}' option '{opt}' has {c.get(opt, 0)} rows")
    if len(train) <= int(customer.get("few_shot", 0)):
        errors.append(f"{len(train)} training rows, not more than few_shot={customer.get('few_shot')}")
    return errors, warnings, {"rows": len(rows), "train": len(train), "holdout": len(hold), "test": len(test),
                              "counts": counts}


def cmd_check(cid):
    customer, rows = P.load_customer(cid)
    errors, warnings, s = validate(customer, rows)
    for w in warnings:
        print(f"warning: {w}")
    for e in errors[:20]:
        print(f"error: {e}")
    if errors:
        die(8, f"customer={cid} {len(errors)} errors (first 20 shown); fix data.jsonl or customer.json")
    print(f"verdict=VALID customer={cid} rows={s['rows']} train={s['train']} holdout={s['holdout']} "
          f"warnings={len(warnings)} next: python3 pipeline/djp.py release {cid}")


# ---------- review: unlabelled rows -> a CSV for a person ----------

def cmd_review(cid, path, state_col, share, teacher, out):
    """Label unlabelled rows with a model, then flag the rows a person should check:
      - every row that will land in the held-out sets (a random ~30 %): they become the ground truth the gate and the
        test score use, and they are the random sample in which new intents show up;
      - the least-confident `share` of the remaining rows: correcting these fixes the most label errors
        (results/review-sweep-2026-09-26.md).
    Writes a CSV (review, reason, state, one column per question, confidence); the person edits the answer columns on
    review=yes rows, then runs `djp.py label <c> <file> --reviewed`. Reads only; no release changes."""
    customer, rows = P.load_customer(cid)
    qs = customer["questions"]
    have = {r["state"] for r in rows}
    texts = []
    for r in read_table(path):
        t = str(r.get(state_col) or "").strip()
        if t and t not in have and t not in texts:
            texts.append(t)
    if not texts:
        die(8, f"no new rows in {path} (column '{state_col}'; rows already labelled are skipped)")
    live = P.current_release(cid)
    teacher = teacher or ("release" if live else "gliner")
    if teacher == "release":
        if not live:
            die(2, f"customer={cid} has no live release; pass --teacher gliner or --teacher djev@3")
        probs = []
        for t in texts:
            a = B.answer(URL, live, t)["answers"]
            probs.append({q: a[q]["probabilities"] for q in qs})
    elif teacher == "gliner":
        probs = B.gliner_read(qs, texts, B.cache_for(cid, "gliner"))
    elif teacher.startswith("djev"):
        k = int(teacher.split("@")[1]) if "@" in teacher else 1
        probs = B.djev_read(URL, qs, None, texts, B.cache_for(cid, "djev"), WORKERS, k)
    else:
        die(2, f"--teacher {teacher}: release | gliner | djev | djev@3")
    conf = [min(max(p[q].values()) for q in qs) for p in probs]  # the least sure question decides
    held = [P.heldout(t) for t in texts]
    rest = sorted((i for i in range(len(texts)) if not held[i]), key=lambda i: conf[i])
    low = set(rest[:int(len(rest) * share)])
    reason = ["held-out (ground truth; new-intent sample)" if held[i] else "least confident" if i in low else ""
              for i in range(len(texts))]
    order = sorted(range(len(texts)), key=lambda i: (reason[i] == "", conf[i]))
    out = out or str(P.customer_dir(cid) / f"review-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review", "reason", "state"] + list(qs) + [f"{q}_confidence" for q in qs])
        for i in order:
            w.writerow(["yes" if reason[i] else "no", reason[i], texts[i]]
                       + [max(probs[i][q], key=probs[i][q].get) for q in qs]
                       + [round(max(probs[i][q].values()), 3) for q in qs])
    os.replace(out + ".tmp", out)
    n_check = sum(1 for x in reason if x)
    print(f"verdict=REVIEW_READY customer={cid} rows={len(texts)} to_check={n_check} "
          f"(held-out {sum(held)}, least-confident {len(low)}) teacher={teacher} file={out} "
          f"next: correct the answer columns on review=yes rows (options: "
          + "; ".join(f"{q}: {', '.join(spec['criteria'])}" for q, spec in qs.items())
          + f"), then python3 pipeline/djp.py label {cid} {out} --reviewed")


# ---------- label: new rows in, retrain, gate ----------

def cmd_label(cid, path, state_col, label_cols, arms, force, reviewed=False):
    """Append labelled rows (dedup by state), then re-release. Safe to re-run with the same file: rows already present
    are skipped, and the release is skipped when the live one was built from the same data.
    reviewed: the file comes from `djp.py review`; rows with review=yes were checked by a person (human labels),
    the rest keep the model's answer and are marked source "teacher" (training only, never scored)."""
    customer, rows = load_valid(cid)
    qs = customer["questions"]
    label_cols = label_cols or list(qs)
    have = {r["state"]: r["answers"] for r in rows}
    new, dup, errors = [], 0, []
    for i, r in enumerate(read_table(path), 1):
        answers = r.get("answers") if isinstance(r.get("answers"), dict) else {c: r.get(c) for c in label_cols}
        state = str(r.get(state_col) or "").strip()
        answers = {q: str(a).strip() for q, a in answers.items() if q in qs and a not in (None, "")}
        if not state:
            errors.append(f"row {i}: no '{state_col}'")
            continue
        missing = [q for q in qs if q not in answers]
        bad = [f"{q}='{a}'" for q, a in answers.items() if a not in qs[q]["criteria"]]
        if missing or bad:
            errors.append(f"row {i}: " + ", ".join([f"no answer for '{q}'" for q in missing] +
                                                   [f"{b} is not an option" for b in bad]))
            continue
        if state in have:
            if have[state] == answers:
                dup += 1
            else:
                errors.append(f"row {i}: state already labelled differently ({have[state]}); change it in data.jsonl")
            continue
        have[state] = answers
        row = {"state": state, "answers": answers}
        if reviewed and str(r.get("review", "")).strip().lower() not in ("yes", "y", "true", "1"):
            row["source"] = "teacher"
        new.append(row)
    if errors:
        for e in errors[:20]:
            print(f"error: {e}")
        die(8, f"customer={cid} {len(errors)} bad rows in {path}; nothing appended. Options: {json.dumps({q: s['criteria'] for q, s in qs.items()})}")
    d = P.customer_dir(cid)
    if new:  # one atomic rewrite: the file holds the old rows or old + new, never a torn append
        P.write_atomic(d / "data.jsonl", "".join(json.dumps(r) + "\n" for r in rows + new))
    print(f"verdict=LABELLED customer={cid} added={len(new)} already_present={dup} rows={len(rows) + len(new)}")
    cur = P.current_release(cid)
    cmd_release(cid, arms or (cur or {}).get("arms") or list(B.ARMS), force)


# ---------- release ----------

def metrics(questions, preds, rows, calibration=None):
    """Per question and overall: accuracy, log-loss (nats), expected calibration error (10 bins)."""
    out = {}
    for q in questions:
        hits, nll, conf_hit = 0, 0.0, []
        for pr, row in zip(preds, rows):
            p = P.calibrate(pr[q], (calibration or {}).get(q))
            truth, best = row["answers"][q], max(p, key=p.get)
            hits += best == truth
            nll -= math.log(max(p.get(truth, 0.0), 1e-6))
            conf_hit.append((p[best], best == truth))
        bins = [[c for c in conf_hit if min(int(c[0] * 10), 9) == b] for b in range(10)]
        ece = sum(len(b) * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b)) for b in bins if b)
        n = len(rows)
        picked = {max(P.calibrate(pr[q], (calibration or {}).get(q)).items(), key=lambda kv: kv[1])[0] for pr in preds}
        unused = sorted({row["answers"][q] for row in rows} - picked)  # options in these rows it never predicts
        out[q] = {"accuracy": round(hits / n, 4), "log_loss": round(nll / n, 4), "ece": round(ece / n, 4),
                  "unused": len(unused)}
    out["_mean"] = {k: round(sum(out[q][k] for q in questions) / len(questions), 4) for k in ("accuracy", "log_loss", "ece")}
    out["_n"] = len(rows)
    return out


def fit_bias(options, probs, truths, steps=400, lr=0.5, l2=0.01):
    """Per-option log-prior shift b minimising log-loss of softmax(log p + b) on (probs, truths)."""
    b = {k: 0.0 for k in options}
    for _ in range(steps):
        g = {k: l2 * b[k] for k in options}
        for p, t in zip(probs, truths):
            c = P.calibrate(p, b)
            for k in options:
                g[k] += (c.get(k, 0.0) - (k == t)) / len(probs)
        b = {k: b[k] - lr * g[k] for k in options}
    return {k: round(v, 4) for k, v in b.items()}


def build_profile(customer, train):
    k = int(customer.get("few_shot", 0))
    return {"context": customer.get("context", ""), "examples": train[:k]}, train[k:]


def load_valid(cid):
    customer, rows = P.load_customer(cid)
    errors, _, _ = validate(customer, rows)
    if errors:
        die(8, f"customer={cid} {len(errors)} data errors, first: {errors[0]}. Run: python3 pipeline/djp.py check {cid}")
    return customer, rows


def candidate(name):
    """-> (backend, served with the profile in front, djev reads averaged). base@N / gliner are also the bars.
    Names: base, profile (+cal), their @N forms (N reads over shuffled option orders), gliner(+cal), tiny, mmbert."""
    head = name.split("+")[0]
    if head.startswith(("base", "profile")):
        return "djev", head.startswith("profile"), int(head.split("@")[1]) if "@" in head else 1
    return head, False, 1


GATE_RULE = ("per question: pass = selection-holdout accuracy >= {bar}, log-loss < {bar}, and no more never-predicted "
             "options than {bar}; among passing candidates the highest accuracy wins, lowest log-loss breaks ties; if "
             "none passes the question stays on {bar}; refuse if every question stays on {bar}; the test split is "
             "reported, never used to choose")


def choose(questions, ev, bar):
    """The gate, per question: among candidates with holdout accuracy >= the bar's and log-loss < the bar's, the one
    with the highest accuracy (lowest log-loss breaks ties; then the name, for determinism); if none passes, the
    question stays on the bar. Accuracy decides because a served answer is right or wrong; log-loss still has to
    beat the bar to pass (set 2026-09-26 after lowest-log-loss picked 77.6 % over 81.6 % on moderation·toxic)."""
    plan = {}
    for q in questions:
        b = ev[bar][q]
        ok = [n for n in ev if n != bar and q in ev[n]
              and ev[n][q]["accuracy"] >= b["accuracy"] and ev[n][q]["log_loss"] < b["log_loss"]
              and ev[n][q].get("unused", 0) <= b.get("unused", 0)]
        plan[q] = min(ok, key=lambda n: (-ev[n][q]["accuracy"], ev[n][q]["log_loss"], n)) if ok else bar
    return plan


def evaluate(cid, arms):
    """Every candidate of every arm, scored on the selection holdout (the gate chooses on it) and on the test split
    (reported only). -> dict with ev, test_ev, plan and what a release needs."""
    customer, rows = load_valid(cid)
    train, hold, test = P.split(rows)
    qs = customer["questions"]
    profile, fit_rows = build_profile(customer, train)
    scored = hold + test  # one read of both; sliced below
    states = lambda rs: [r["state"] for r in rs]
    fit = lambda preds, rs: {q: fit_bias(qs[q]["criteria"], [x[q] for x in preds], [r["answers"][q] for r in rs]) for q in qs}
    cands, tiny, models = {}, {}, {}  # cands: name -> (calibration {q: bias}, predictions on scored rows, questions)
    n = B.DJEV_READS
    if "djev" in arms:
        c = B.cache_for(cid, "djev")
        read = lambda prof, rs, k=1: B.djev_read(URL, qs, prof, states(rs), c, WORKERS, k)
        for tag, k in [("", 1)] + ([(f"@{n}", n)] if n > 1 else []):
            base_s, prof_s = read(None, scored, k), read(profile, scored, k)
            cands["base" + tag] = ({}, base_s, list(qs))
            cands[f"base{tag}+cal"] = (fit(read(None, train, k), train), base_s, list(qs))
            cands["profile" + tag] = ({}, prof_s, list(qs))
            cands[f"profile{tag}+cal"] = (fit(read(profile, fit_rows, k), fit_rows), prof_s, list(qs))
    if "gliner" in arms:
        c = B.cache_for(cid, "gliner")
        g_s = B.gliner_read(qs, states(scored), c)
        cands["gliner"] = ({}, g_s, list(qs))
        cands["gliner+cal"] = (fit(B.gliner_read(qs, states(train), c), train), g_s, list(qs))
    for arm in [a for a in B.LOCAL_TRAINED if a in arms]:
        t_s, done = [{} for _ in scored], []
        for q, spec in qs.items():
            ok, least = B.tiny_eligible(spec, q, rows)
            tiny.setdefault(q, {"eligible": ok, "least_per_option": least, "min_per_option": B.TINY_MIN_PER_OPTION})
            if not ok:
                continue
            if arm == "tfidf":  # CPU, seconds
                clf, secs = B.tfidf_fit(q, spec, train)
                probs = B.tfidf_proba(clf, states(scored), spec["criteria"])
                models[(arm, q)] = clf
            else:
                with B.gpu_lock():  # train -> score -> park: one model on the GPU per machine
                    m, tok, secs = B.tiny_fit(q, spec, train, arm)
                    probs = B.tiny_proba(m, tok, states(scored), arm)
                    models[(arm, q)] = (B.park(m), tok)  # kept on CPU until saved
            for d, p in zip(t_s, probs):
                d[q] = p
            tiny[q][f"{arm}_train_s"] = round(secs, 1)
            done.append(q)
        if done:
            cands[arm] = ({}, t_s, done)
    bar = (f"base@{n}" if n > 1 else "base") if "djev" in arms else "gliner" if "gliner" in arms else None
    if bar is None:
        die(2, f"--arms {','.join(arms)} has no zero-shot bar; include djev (bar: plain djev) or gliner (bar: plain gliner)")
    h = len(hold)
    ev = {n: metrics(qs_n, p[:h], hold, cal) for n, (cal, p, qs_n) in cands.items()}
    test_ev = {n: metrics(qs_n, p[h:], test, cal) for n, (cal, p, qs_n) in cands.items()} if test else {}
    plan = choose(qs, ev, bar)
    return {"customer": customer, "qs": qs, "train": train, "hold": hold, "test": test, "profile": profile,
            "fit_rows": fit_rows, "cands": cands, "ev": ev, "test_ev": test_ev, "bar": bar, "plan": plan, "tiny": tiny,
            "models": models}


def cmd_eval(cid, arms):
    """Print every candidate's holdout accuracy / log-loss per question, and what the gate would pick. Writes nothing
    but the read caches."""
    e = evaluate(cid, arms)
    print(f"customer={cid} holdout n={len(e['hold'])} bar={e['bar']} arms={','.join(arms)}")
    for q in e["qs"]:
        cells = " ".join(f"{n}={e['ev'][n][q]['accuracy']}/{e['ev'][n][q]['log_loss']}" for n in e["ev"] if q in e["ev"][n])
        print(f"  {q}: pick={e['plan'][q]}  acc/ll: {cells}")


def _lock(cid):
    """Exclusive per-customer lock (released on close): two label/release/rollback runs never interleave."""
    import fcntl
    f = open(P.customer_dir(cid) / ".lock", "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def cmd_rollback(cid, version=None):
    """Point CURRENT at an earlier release that passed its gate (default: the one before the live one)."""
    rel = P.customer_dir(cid) / "releases"
    log = [json.loads(l) for l in (rel / "log.jsonl").read_text().splitlines()] if (rel / "log.jsonl").exists() else []
    passed = sorted({x["version"] for x in log if x["passed"] and "rollback_from" not in x
                     and (rel / x["version"] / "release.json").exists()})
    cur = (rel / "CURRENT").read_text().strip() if (rel / "CURRENT").exists() else None
    if version is None:
        older = [v for v in passed if cur is None or v < cur]
        version = older[-1] if older else None
    if version not in passed:
        die(2, f"customer={cid} no passed release {version or 'before ' + str(cur)} to roll back to; passed: {passed[-5:]}")
    P.write_atomic(rel / "CURRENT", version + "\n")
    with open(rel / "log.jsonl", "a") as f:
        f.write(json.dumps({"version": version, "rollback_from": cur, "passed": True,
                            "at": time.strftime("%Y%m%d-%H%M%S")}) + "\n")
    print(f"verdict=ROLLED_BACK customer={cid} live={version} was={cur}")


def cmd_release(cid, arms, force=False):
    t0 = time.time()
    sha = P.data_sha(cid)
    cur = P.current_release(cid)
    if (cur and not force and cur.get("data_sha") == sha and cur.get("arms") == list(arms)
            and ("djev" not in arms or cur.get("djev_reads", 1) == B.DJEV_READS)
            and cur.get("gate", {}).get("rule") == GATE_RULE.format(bar=cur.get("bar", "base"))):
        print(f"verdict=UNCHANGED customer={cid} live={cur['version']} built from the same data, arms and gate rule "
              f"(data_sha={sha}); add --force to rebuild")
        return
    e = evaluate(cid, arms)
    qs, ev, plan, bar, cands, hold = e["qs"], e["ev"], e["plan"], e["bar"], e["cands"], e["hold"]
    per_q = {q: ev[plan[q]][q] for q in qs}
    mean = {k: round(sum(per_q[q][k] for q in qs) / len(qs), 4) for k in ("accuracy", "log_loss", "ece")}
    passed = any(v != bar for v in plan.values())
    d = P.customer_dir(cid)
    rel = d / "releases"
    version = time.strftime("%Y%m%d-%H%M%S")
    while (rel / version).exists():
        version += "a"
    rplan = {}
    for q in qs:
        backend, use_profile, reads = candidate(plan[q])
        rplan[q] = {"variant": plan[q], "backend": backend, "use_profile": use_profile, "reads": reads,
                    "calibration": cands[plan[q]][0].get(q, {})}
        if backend in B.LOCAL_TRAINED:
            rplan[q]["path"] = B.tiny_dirname(q, backend)
    release = {"customer": cid, "version": version, "kind": "backends", "arms": list(arms), "bar": bar, "data_sha": sha,
               "djev_reads": B.DJEV_READS,
               "base": {"image": IMAGE, "model": B.DJEV_MODEL}, "gliner_model": B.GLINER_MODEL,
               "trained_models": {a: m for a, (m, _, _) in B.TRAINED.items()}, "questions": qs, "profile": e["profile"],
               "plan": rplan, "tiny": e["tiny"],
               "split": {"train": len(e["train"]), "few_shot": len(e["profile"]["examples"]),
                         "calibration_rows": len(e["fit_rows"]), "holdout": len(hold), "test": len(e["test"])},
               "eval": ev, "released_eval": {**per_q, "_mean": mean},
               "test_eval": {"n": len(e["test"]), "bar": {q: e["test_ev"][bar][q] for q in qs} if e["test_ev"] else {},
                             "released": {q: e["test_ev"][plan[q]][q] for q in qs} if e["test_ev"] else {}},
               "gate": {"passed": passed,
                        "rule": GATE_RULE.format(bar=bar)},
               "seconds": round(time.time() - t0, 1)}
    # build in a scratch dir, rename into place, then move CURRENT: a crash leaves the old release live
    for old in rel.glob(".tmp-*"):
        shutil.rmtree(old, ignore_errors=True)
    tmp = rel / f".tmp-{version}"
    if passed:
        for q, spec in rplan.items():
            if spec["backend"] == "tfidf":
                B.tfidf_save(e["models"][("tfidf", q)], str(tmp / spec["path"]))
            elif spec["backend"] in B.TRAINED:
                B._tiny().save(*e["models"][(spec["backend"], q)], str(tmp / spec["path"]))
    P.write_atomic(tmp / "release.json", json.dumps(release, indent=1))
    os.replace(tmp, rel / version)
    b = ev[bar]["_mean"]
    line = (f"customer={cid} version={version} holdout n={len(hold)} bar={bar} acc={b['accuracy']} ll={b['log_loss']}"
            f" -> acc={mean['accuracy']} ll={mean['log_loss']}"
            + "".join(f" {q}:{plan[q]}={per_q[q]['accuracy']}({bar} {ev[bar][q]['accuracy']})" for q in qs)
            + (f" test n={len(e['test'])}: " + " ".join(
                f"{q}={release['test_eval']['released'][q]['accuracy']}({bar} {release['test_eval']['bar'][q]['accuracy']})"
                for q in qs) if e["test_ev"] else ""))
    with open(rel / "log.jsonl", "a") as f:
        f.write(json.dumps({"version": version, "passed": passed, "data_sha": sha, "arms": list(arms), "bar": bar,
                            "holdout": len(hold), "test": len(e["test"]), "plan": {q: plan[q] for q in qs},
                            "accuracy": {q: per_q[q]["accuracy"] for q in qs}, "seconds": release["seconds"]}) + "\n")
    if not passed:
        die(7, line + f" (nothing beats {bar}; record kept at {rel / version}/release.json; CURRENT unchanged)")
    P.write_atomic(rel / "CURRENT", version + "\n")
    print("verdict=RELEASED " + line + f" seconds={release['seconds']} serve=/c/{cid}/v1/systemone")


def cmd_status(as_json=False):
    out = []
    for d in P.all_customers():
        r = P.current_release(d.name)
        s = {"customer": d.name, "where": "example" if d.parent == P.EXAMPLES else "home", "live": None}
        if r:
            bar = r.get("bar", "base")
            s.update({"live": r["version"], "kind": r["kind"], "n": r["split"]["holdout"],
                      "holdout_acc": r["released_eval"]["_mean"]["accuracy"],
                      "bar_acc": r["eval"][bar]["_mean"]["accuracy"], "bar": bar,
                      "stale": r.get("data_sha") != P.data_sha(d.name),
                      "test_n": r.get("test_eval", {}).get("n", 0),
                      "test_acc": round(sum(v["accuracy"] for v in r["test_eval"]["released"].values())
                                        / len(r["test_eval"]["released"]), 4) if r.get("test_eval", {}).get("released") else None,
                      "plan": {q: p.get("variant") for q, p in r["plan"].items()}})
        out.append(s)
    if as_json:
        print(json.dumps(out))
        return
    for s in out:
        if not s["live"]:
            print(f"customer={s['customer']} ({s['where']}) live=none")
            continue
        print(f"customer={s['customer']} ({s['where']}) live={s['live']} holdout_acc={s['holdout_acc']} "
              f"{s['bar']}_acc={s['bar_acc']} n={s['n']} test_acc={s['test_acc']} test_n={s['test_n']} "
              f"stale={str(s['stale']).lower()} "
              + " ".join(f"{q}={v}" for q, v in s["plan"].items()))


def flags(args):
    """--key value pairs and bare --force. -> (positional, {key: value})."""
    pos, opt, i = [], {}, 0
    while i < len(args):
        if args[i] in ("--force", "--json", "--reviewed"):
            opt[args[i]] = True
            i += 1
        elif args[i].startswith("--") and i + 1 < len(args):
            opt[args[i]] = args[i + 1]
            i += 2
        else:
            pos.append(args[i])
            i += 1
    return pos, opt


def parse_arms(opt, default=None):
    if "--arms" not in opt:
        return default
    arms = [a.strip() for a in opt["--arms"].split(",") if a.strip()]
    bad = [a for a in arms if a not in B.ARMS]
    if bad or not arms:
        die(2, f"--arms {opt['--arms']}: choose from {','.join(B.ARMS)}")
    return arms


def run(cmd, cid, args, opt, arms):
    if cmd == "import":
        if len(args) < 2 or "--state" not in opt or "--label" not in opt:
            die(2, "usage: djp.py import <customer> <file.csv|.jsonl> --state <col> --label <col>[,<col>...]")
        cmd_import(cid, args[1], opt["--state"], [c.strip() for c in opt["--label"].split(",") if c.strip()])
    elif cmd == "label":
        if len(args) < 2:
            die(2, "usage: djp.py label <customer> <file.csv|.jsonl> [--state <col>] [--label <cols>] [--arms ...] [--force]")
        cols = [c.strip() for c in opt["--label"].split(",") if c.strip()] if "--label" in opt else None
        cmd_label(cid, args[1], opt.get("--state", "state"), cols, arms, opt.get("--force", False),
                  opt.get("--reviewed", False))
    elif cmd == "review":
        if len(args) < 2:
            die(2, "usage: djp.py review <customer> <unlabelled.csv|.jsonl> [--state state] [--share 0.5] "
                   "[--teacher release|gliner|djev@3] [--out file.csv]")
        cmd_review(cid, args[1], opt.get("--state", "state"), float(opt.get("--share", 0.5)),
                   opt.get("--teacher"), opt.get("--out"))
    elif cmd == "check":
        cmd_check(cid)
    elif cmd == "release":
        cmd_release(cid, arms or list(B.ARMS), opt.get("--force", False))
    elif cmd == "rollback":
        cmd_rollback(cid, args[1] if len(args) > 1 else None)
    else:
        cmd_eval(cid, arms or list(B.ARMS))


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 2
    cmd, (args, opt) = argv[0], flags(argv[1:])
    try:
        if cmd == "status":
            return cmd_status(opt.get("--json", False)) or 0
        if not args:
            die(2, f"usage: djp.py {cmd} <customer>")
        cid = args[0]
        if not cid.replace("-", "").replace("_", "").isalnum():
            die(2, f"customer id '{cid}': letters, digits, - and _ only")
        if cmd == "init":
            return cmd_init(cid) or 0
        if P.customer_dir(cid) is None:
            die(2, f"no customer '{cid}' in {P.HOME} or {P.EXAMPLES}. Run: python3 pipeline/djp.py init {cid}")
        lock = _lock(cid) if cmd in ("label", "release", "rollback") else None  # review only reads and writes its CSV
        try:
            run(cmd, cid, args, opt, parse_arms(opt))
        finally:
            if lock:
                lock.close()
    except B.NeedsVenv as e:
        die(2, f"{e}. Run with .venv/bin/python, or pass --arms djev")
    except urllib.error.HTTPError as e:
        die(2, f"{URL} answered HTTP {e.code}: {e.read()[:200]!r}. Logs: gcloud run services logs read djev --region us-central1 --limit 50")
    except OSError as e:
        die(2, f"cannot reach {URL} ({e}). Start a /v1/systemone server there (for Cloud Run: gcloud run services proxy <service> --port 8081) "
               f"(reads done so far are cached; re-run the same command), or pass --arms gliner,tiny,mmbert")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
