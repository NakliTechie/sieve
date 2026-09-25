#!/usr/bin/env python3
"""djp: onboard a customer, then build, gate and release their own djev deployment.

  python3 pipeline/djp.py status                               # one line per customer: live release, holdout vs base
  python3 pipeline/djp.py init <customer>                      # create $DJP_HOME/<customer>/customer.json to edit
  python3 pipeline/djp.py import <customer> <file.csv|.jsonl> --state <col> --label <col>[,<col>...]
                                                               # their data, their column names -> data.jsonl
  python3 pipeline/djp.py check <customer>                     # validate data + questions; no GPU calls
  python3 pipeline/djp.py release <customer>                   # eval base -> candidates -> per-question gate -> release
  python3 pipeline/djp.py eval <customer> [base|profile|calibrated]   # holdout metrics for one variant

Customer data lives in $DJP_HOME (default data/customers, gitignored). Layer 1 (this file): a release is the
customer's rules + examples (profile) + per-option calibration, served on the shared model at
/c/<customer>/v1/systemone. Layer 2 (TRAINING.md) swaps the profile for trained weights behind the same gate.

Env: DJP_HOME, DJEV_URL (default http://localhost:8081, the `gcloud run services proxy`), DJEV_WORKERS (default 8).
Exit: 0 ok · 2 setup (customer missing, service unreachable) · 7 gate refused (nothing released) · 8 data invalid.
"""
import csv
import json
import math
import os
import sys
import time
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djcore as P  # noqa: E402

URL = os.environ.get("DJEV_URL", "http://localhost:8081")
WORKERS = int(os.environ.get("DJEV_WORKERS", "8"))
IMAGE = "ghcr.io/taeold/djev-run@sha256:a352b97ab4ccf9c0d5caabbb7df16e0b40a801426cd8582c0e3fe81b2da16ac1"
MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4@ec4ff3df205028f4e81c954c2227f9312b3ec2ea"
MIN_HOLDOUT = 50  # held-out rows per question before a release's numbers mean much
VERDICTS = {2: "SETUP", 7: "GATE_REFUSED", 8: "DATA_INVALID"}
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
    train, hold = P.split(rows) if rows else ([], [])
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
    return errors, warnings, {"rows": len(rows), "train": len(train), "holdout": len(hold), "counts": counts}


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


# ---------- release ----------

def predict(questions, profile, rows):
    """-> list of {question: {option: p}} for rows, raw model probabilities (no calibration)."""
    def one(row):
        a = P.ask(URL, questions, P.render_state(profile, row["state"]))["answers"]
        return {q: a[q]["probabilities"] for q in questions}
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(one, rows))


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
        out[q] = {"accuracy": round(hits / n, 4), "log_loss": round(nll / n, 4), "ece": round(ece / n, 4)}
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


def cmd_eval(cid, variant="calibrated"):
    customer, rows = load_valid(cid)
    train, hold = P.split(rows)
    profile, fit_rows = build_profile(customer, train)
    qs = customer["questions"]
    cal = None
    if variant == "base":
        profile = None
    elif variant == "calibrated":
        fp = predict(qs, profile, fit_rows)
        cal = {q: fit_bias(qs[q]["criteria"], [x[q] for x in fp], [r["answers"][q] for r in fit_rows]) for q in qs}
    print(json.dumps(metrics(qs, predict(qs, profile, hold), hold, cal), indent=1))


def cmd_release(cid):
    t0 = time.time()
    customer, rows = load_valid(cid)
    train, hold = P.split(rows)
    qs = customer["questions"]
    profile, fit_rows = build_profile(customer, train)
    fit = lambda preds, rs: {q: fit_bias(qs[q]["criteria"], [x[q] for x in preds], [r["answers"][q] for r in rs]) for q in qs}
    base_train, base_hold = predict(qs, None, train), predict(qs, None, hold)
    prof_fit, prof_hold = predict(qs, profile, fit_rows), predict(qs, profile, hold)
    # candidates: (profile or not) x (calibration or not); base itself is the bar, never a release
    cands = {"base+cal": (None, fit(base_train, train), base_hold),
             "profile": (profile, {}, prof_hold),
             "profile+cal": (profile, fit(prof_fit, fit_rows), prof_hold)}
    ev = {"base": metrics(qs, base_hold, hold)}
    ev.update({name: metrics(qs, h, hold, cal) for name, (_, cal, h) in cands.items()})
    base = ev["base"]
    # per question: the variant with the lowest holdout log-loss among those that keep accuracy >= base;
    # a question with no such variant stays on the plain base model
    plan = {}
    for q in qs:
        ok = [n for n in cands if ev[n][q]["accuracy"] >= base[q]["accuracy"] and ev[n][q]["log_loss"] < base[q]["log_loss"]]
        plan[q] = min(ok, key=lambda n: ev[n][q]["log_loss"]) if ok else "base"
    per_q = {q: (base[q] if plan[q] == "base" else ev[plan[q]][q]) for q in qs}
    mean = {k: round(sum(per_q[q][k] for q in qs) / len(qs), 4) for k in ("accuracy", "log_loss", "ece")}
    passed = any(v != "base" for v in plan.values())
    version = time.strftime("%Y%m%d-%H%M%S")
    release = {"customer": cid, "version": version, "kind": "profile", "base": {"image": IMAGE, "model": MODEL},
               "questions": qs, "profile": profile,
               "plan": {q: {"variant": plan[q], "use_profile": plan[q].startswith("profile"),
                            "calibration": cands[plan[q]][1].get(q, {}) if plan[q] != "base" else {}} for q in qs},
               "split": {"train": len(train), "few_shot": len(profile["examples"]), "calibration_rows": len(fit_rows),
                         "holdout": len(hold)},
               "eval": ev, "released_eval": {**per_q, "_mean": mean},
               "gate": {"passed": passed,
                        "rule": "per question: holdout accuracy >= base and log-loss < base, else that question stays on base; refuse if every question stays on base"},
               "seconds": round(time.time() - t0, 1)}
    d = P.customer_dir(cid)
    rdir = d / "releases" / version
    P.write_atomic(rdir / "release.json", json.dumps(release, indent=1))
    b = base["_mean"]
    line = (f"customer={cid} version={version} holdout n={len(hold)} base acc={b['accuracy']} ll={b['log_loss']}"
            f" -> acc={mean['accuracy']} ll={mean['log_loss']}"
            + "".join(f" {q}:{plan[q]}={per_q[q]['accuracy']}(base {base[q]['accuracy']})" for q in qs))
    if not passed:
        die(7, line + f" (nothing beats base; record kept at {rdir}/release.json; CURRENT unchanged)")
    P.write_atomic(d / "releases" / "CURRENT", version + "\n")
    print("verdict=RELEASED " + line + f" serve=/c/{cid}/v1/systemone")


def cmd_status():
    for d in P.all_customers():
        r = P.current_release(d.name)
        where = "example" if d.parent == P.EXAMPLES else "home"
        if not r:
            print(f"customer={d.name} ({where}) live=none")
            continue
        m = r["released_eval"]["_mean"]
        print(f"customer={d.name} ({where}) live={r['version']} kind={r['kind']} holdout_acc={m['accuracy']} "
              f"base_acc={r['eval']['base']['_mean']['accuracy']} n={r['split']['holdout']} serve=/c/{d.name}/v1/systemone")


def main(argv):
    cmds = ("status", "init", "import", "check", "release", "eval")
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    try:
        if cmd == "status":
            return cmd_status() or 0
        if not args:
            die(2, f"usage: djp.py {cmd} <customer>")
        cid = args[0]
        if not cid.replace("-", "").replace("_", "").isalnum():
            die(2, f"customer id '{cid}': letters, digits, - and _ only")
        if cmd == "init":
            return cmd_init(cid) or 0
        if P.customer_dir(cid) is None:
            die(2, f"no customer '{cid}' in {P.HOME} or {P.EXAMPLES}. Run: python3 pipeline/djp.py init {cid}")
        if cmd == "import":
            opt = {args[i]: args[i + 1] for i in range(2, len(args) - 1, 2)}
            if len(args) < 2 or "--state" not in opt or "--label" not in opt:
                die(2, "usage: djp.py import <customer> <file.csv|.jsonl> --state <col> --label <col>[,<col>...]")
            cmd_import(cid, args[1], opt["--state"], [c.strip() for c in opt["--label"].split(",") if c.strip()])
        elif cmd == "check":
            cmd_check(cid)
        elif cmd == "release":
            cmd_release(cid)
        else:
            cmd_eval(cid, args[1] if len(args) > 1 else "calibrated")
    except urllib.error.HTTPError as e:
        die(2, f"{URL} answered HTTP {e.code}: {e.read()[:200]!r}. Logs: gcloud run services logs read djev --region <region> --limit 50")
    except OSError as e:
        die(2, f"cannot reach {URL} ({e}). Start: gcloud run services proxy djev --region <region> --port 8081")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
