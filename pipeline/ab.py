#!/usr/bin/env python3
"""A/B: djev (/v1/systemone) vs GLiNER2.5-Decide on the same rows, questions and options, plus an option-order test.

  .venv/bin/python pipeline/ab.py run [--arms djev,gliner] [--orders orig,again,rev,shuf] [--limit N] [--run NAME]
                                    [--reads K] [--source customer|fast-decisions] [--max-seconds S]
  python3 pipeline/ab.py report [run-dir]          # aggregates -> stdout (markdown); default: newest run in data/ab

Suite: the held-out rows of every customer in $DJP_HOME and examples (djcore.split), plus the single-choice questions of
fastino/fast-decisions (data/bench/fast-decisions/*.jsonl, Apache-2.0). Both arms get the same information:
question name + options. No customer rules, no few-shot, no calibration: this compares base models.

Order sub-test: each row runs with options in the original order, the original order again (the noise floor: djev
randomises its answer slots per call), reversed, and shuffled (seeded per row).
Reported: accuracy per order, flip rate (answer changes when only the order changes), and position bias (how often
the first / last option shown is picked, against how often it is the right answer).

Permutation averaging (--reads K): each (row, order) is read K times, read 0 in that order and reads 1..K-1 in
shuffled orders; `report` then scores the average of the first k reads for every k <= K (flip rate, accuracy, cost).

Results append to data/ab/<run>/results.jsonl as they arrive; re-running `run` with the same --run skips what is done
(per read, so --reads 5 after --reads 3 adds only reads 3 and 4). --max-seconds stops cleanly (exit 3, resumable).
Env: DJEV_URL (default http://localhost:8081), DJEV_WORKERS (default 8), GLINER_MODEL (default fastino/GLiNER2.5-Decide).
Records before 2026-09-26 carry picks only (no probabilities, no `read`); they count as read 0 in `report`.
"""
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import djcore as P  # noqa: E402

URL = os.environ.get("DJEV_URL", "http://localhost:8081")
WORKERS = int(os.environ.get("DJEV_WORKERS", "8"))
BENCH = P.DATA / "bench" / "fast-decisions"
AB = P.DATA / "ab"


def suite(limit=None):
    """-> items {id, source, domain, text, questions {q: [options]}, truth {q: label}}."""
    items = []
    for d in P.all_customers():
        customer, rows = P.load_customer(d.name)
        if not rows:
            continue
        _, hold, _ = P.split(rows)
        qs = {q: list(s["criteria"]) for q, s in customer["questions"].items()}
        for i, r in enumerate(hold[:limit]):
            items.append({"id": f"c:{d.name}:{i}", "source": "customer", "domain": d.name, "text": r["state"],
                          "questions": qs, "truth": {q: r["answers"][q] for q in qs}})
    for f in sorted(BENCH.glob("*.jsonl")):
        for i, line in enumerate(f.read_text().splitlines()[:limit]):
            r = json.loads(line)
            cls = [c for c in r["output"]["classifications"] if not c["multi_label"]]
            if cls:
                items.append({"id": f"fd:{f.stem}:{i}", "source": "fast-decisions", "domain": f.stem, "text": r["input"],
                              "questions": {c["task"]: list(c["labels"]) for c in cls},
                              "truth": {c["task"]: c["true_label"][0] for c in cls}})
    return items


def ordered(item, order):
    if order in ("orig", "again"):  # "again" repeats the call unchanged: the noise floor for flips
        return item["questions"]
    if order == "rev":
        return {q: o[::-1] for q, o in item["questions"].items()}
    rng = random.Random(item["id"])
    return {q: rng.sample(o, len(o)) for q, o in item["questions"].items()}


def arm_djev(text, qs):
    """-> ({q: {option: p}}, round-trip ms, server ms)."""
    t0 = time.time()
    r = P.ask(URL, {q: {"type": "choice", "instructions": q.replace("_", " "), "criteria": o} for q, o in qs.items()}, text)
    return {q: r["answers"][q]["probabilities"] for q in qs}, round((time.time() - t0) * 1000, 1), \
        r.get("diagnostics", {}).get("timing", {}).get("total_ms")


def arm_gliner(text, qs):
    t0 = time.time()
    probs = B.gliner_read({q: {"criteria": o} for q, o in qs.items()}, [text])[0]
    ms = round((time.time() - t0) * 1000, 1)
    return probs, ms, ms


ARMS = {"djev": (arm_djev, WORKERS), "gliner": (arm_gliner, 1)}


def permuted(item, order, i):
    """Read 0 shows the options in `order`; read i > 0 in an order shuffled by (item, order, i), independent across
    orders, so averaging over reads is compared fairly between orig and rev."""
    qs = ordered(item, order)
    if i == 0:
        return qs
    rng = random.Random(f"{item['id']}|{order}|{i}")
    return {q: rng.sample(o, len(o)) for q, o in qs.items()}


def cmd_run(args):
    opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
    arms = opt.get("--arms", "djev,gliner").split(",")
    orders = opt.get("--orders", "orig,again,rev,shuf").split(",")
    reads = int(opt.get("--reads", 1))
    limit = int(opt["--limit"]) if "--limit" in opt else None
    budget_s = float(opt.get("--max-seconds", 0)) or None  # stop cleanly (resumable) after this long
    run = AB / opt.get("--run", time.strftime("%Y%m%d"))
    run.mkdir(parents=True, exist_ok=True)
    out = run / "results.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue  # a line torn by a kill; that read is redone
            if "error" not in x:
                done.add((x["arm"], x["order"], x["id"], x.get("read", 0)))
    items = [it for it in suite(limit) if opt.get("--source", it["source"]) == it["source"]]
    lock = threading.Lock()
    t_start = time.time()
    for arm in arms:
        fn, workers = ARMS[arm]
        todo = [(o, it, i) for i in range(reads) for o in orders for it in items if (arm, o, it["id"], i) not in done]
        print(f"arm={arm} items={len(items)} orders={orders} reads={reads} todo={len(todo)}", flush=True)
        t0, n, stop = time.time(), [0], threading.Event()

        def one(job):
            if stop.is_set():
                return
            if budget_s and time.time() - t_start > budget_s:
                stop.set()
                return
            order, it, i = job
            qs = permuted(it, order, i)
            base = {"arm": arm, "order": order, "read": i, "id": it["id"], "source": it["source"], "domain": it["domain"]}
            try:
                probs, ms, server_ms = fn(it["text"], qs)
                rec = {**base, "shown": qs, "truth": it["truth"], "probs": probs,
                       "pick": {q: max(p, key=p.get) for q, p in probs.items()}, "ms": ms, "server_ms": server_ms}
            except Exception as e:  # recorded, not fatal: one bad row must not lose a long run
                rec = {**base, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n[0] += 1
                if n[0] % 500 == 0:
                    print(f"  {arm}: {n[0]}/{len(todo)} in {round(time.time() - t0)} s", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        secs = round(time.time() - t0, 1)
        with open(run / "meta.jsonl", "a") as f:
            f.write(json.dumps({"arm": arm, "calls": n[0], "seconds": secs, "workers": workers, "reads": reads,
                                "at": time.strftime("%Y%m%d-%H%M%S")}) + "\n")
        print(f"arm={arm} finished {n[0]} in {secs} s" + (" (stopped at --max-seconds)" if stop.is_set() else ""), flush=True)
        if stop.is_set():
            print(f"verdict=STOPPED run={run} budget reached; re-run the same command to resume")
            return 3
    print(f"verdict=DONE run={run} next: python3 pipeline/ab.py report {run}")


def pct(x):
    return f"{100 * x:.1f}%"


def cmd_report(args):
    # default: the most recently written run (not the alphabetically last, which picked `smoke` over `full`)
    run = Path(args[0]) if args else max((d for d in AB.iterdir() if (d / "results.jsonl").exists()),
                                         key=lambda d: (d / "results.jsonl").stat().st_mtime)
    recs = []
    for line in (run / "results.jsonl").read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    allok = [r for r in recs if "error" not in r]
    ok = [r for r in allok if r.get("read", 0) == 0]
    errs_all = [r for r in recs if "error" in r]
    recs = [r for r in recs if r.get("read", 0) == 0]
    errs = [r for r in recs if "error" in r]
    arms = sorted({r["arm"] for r in ok})
    orders = [o for o in ("orig", "again", "rev", "shuf") if any(r["order"] == o for r in ok)]
    print(f"# A/B run {run.name}\n\n{len(ok)} scored calls, {len(errs)} errors. Accuracy is exact match per question.\n")
    # accuracy by source x arm x order
    print("## Accuracy\n\n| Rows | Questions | " + " | ".join(f"{a} {o}" for a in arms for o in orders) + " |")
    print("|---|---|" + "---|" * (len(arms) * len(orders)))
    groups = [("fast-decisions (all 17 domains)", lambda r: r["source"] == "fast-decisions")] + \
             [(f"customer {d}", (lambda d: lambda r: r["domain"] == d)(d))
              for d in sorted({r["domain"] for r in ok if r["source"] == "customer"})] + \
             [(f"fd {d}", (lambda d: lambda r: r["domain"] == d)(d))
              for d in sorted({r["domain"] for r in ok if r["source"] == "fast-decisions"})]
    for name, g in groups:
        cells, nq = [], 0
        for a in arms:
            for o in orders:
                rs = [r for r in ok if r["arm"] == a and r["order"] == o and g(r)]
                hits = [r["pick"][q] == r["truth"][q] for r in rs for q in r["truth"]]
                nq = max(nq, len(hits))
                cells.append(pct(sum(hits) / len(hits)) if hits else "–")
        print(f"| {name} | {nq} | " + " | ".join(cells) + " |")
    # order sensitivity
    print("\n## Option order\n\nFlip rate: share of answers that change when only the order of options changes."
          " First/last: how often the option shown first/last is picked, minus how often it is the right answer"
          " (0 = no position bias). Questions with 3+ options only for first/last.\n")
    print("| Arm | Flip, same order again (noise) | Flip vs reversed | Flip vs shuffled | First picked − first right | Last picked − last right |")
    print("|---|---|---|---|---|---|")
    for a in arms:
        by = {(r["order"], r["id"]): r for r in ok if r["arm"] == a}
        flips = {}
        for o in ("again", "rev", "shuf"):
            pairs = [(by[("orig", i)], by[(o, i)]) for (oo, i) in by if oo == o and ("orig", i) in by]
            f = [x["pick"][q] != y["pick"][q] for x, y in pairs for q in x["truth"]]
            flips[o] = pct(sum(f) / len(f)) if f else "–"
        fp = fr = lp = lr = n = 0
        for r in by.values():
            if r["order"] == "again":
                continue
            for q, shown in r["shown"].items():
                if len(shown) < 3:
                    continue
                n += 1
                fp += r["pick"][q] == shown[0]
                fr += r["truth"][q] == shown[0]
                lp += r["pick"][q] == shown[-1]
                lr += r["truth"][q] == shown[-1]
        bias = (lambda p, t: f"{100 * (p - t) / n:+.1f} pts") if n else (lambda p, t: "–")
        print(f"| {a} | {flips['again']} | {flips['rev']} | {flips['shuf']} | {bias(fp, fr)} | {bias(lp, lr)} |")
    # latency
    print("\n## Latency per call (all questions of a row in one call)\n\n| Arm | Server p50 | Server p95 | Round trip p50 |\n|---|---|---|---|")
    for a in arms:
        s = sorted(r["server_ms"] for r in ok if r["arm"] == a and r.get("server_ms"))
        m = sorted(r["ms"] for r in ok if r["arm"] == a)
        q = lambda xs, p: f"{xs[min(int(len(xs) * p), len(xs) - 1)]:.0f} ms" if xs else "–"
        print(f"| {a} | {q(s, .5)} | {q(s, .95)} | {q(m, .5)} |")
    if any(r.get("read", 0) > 0 for r in allok):
        perm_report(run, allok)
    if errs_all:
        errs = errs_all
    if errs:
        print("\n## Errors\n")
        for e in sorted({(r['arm'], r['domain'], r['error']) for r in errs})[:10]:
            print(f"- {e[0]} {e[1]}: {e[2]}")


def perm_report(run, recs):
    """Average of the first k reads (shuffled option orders) for every k: accuracy per order, flip rates, cost."""
    by = {}
    for r in recs:
        by.setdefault((r["arm"], r["order"], r["id"]), {})[r.get("read", 0)] = r
    meta = [json.loads(l) for l in (run / "meta.jsonl").read_text().splitlines()] if (run / "meta.jsonl").exists() else []
    print("\n## Permutation averaging (average of k reads over shuffled option orders)\n")
    print("Each row is the same decisions scored on the average of reads 0..k-1. Only (row, order) pairs with all k "
          "reads count. Flip: answer changes against the original order. Cost: calls per decision = k.\n")
    print("| Arm | Rows | k | Decisions | Acc orig | Acc again | Acc rev | Acc shuf | Flip again (noise) | Flip rev | Flip shuf |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in sorted({k[0] for k in by}):
        for src in ("fast-decisions", "customer"):
            kmax = max((len(v) for (a, o, i), v in by.items() if a == arm and v and next(iter(v.values()))["source"] == src),
                       default=0)
            for k in range(1, kmax + 1):
                picks = {}
                for (a, o, i), v in by.items():
                    if a != arm or not all(j in v for j in range(k)) or v[0]["source"] != src or "probs" not in v[0]:
                        continue
                    avg = {q: {opt: sum(v[j]["probs"][q].get(opt, 0) for j in range(k)) / k for opt in v[0]["probs"][q]}
                           for q in v[0]["probs"]}
                    picks[(o, i)] = ({q: max(p, key=p.get) for q, p in avg.items()}, v[0]["truth"])
                if not picks:
                    continue
                acc, flip = {}, {}
                for o in ("orig", "again", "rev", "shuf"):
                    hits = [pk[q] == t[q] for (oo, _), (pk, t) in picks.items() if oo == o for q in t]
                    acc[o] = pct(sum(hits) / len(hits)) if hits else "–"
                    if o != "orig":
                        f = [picks[(o, i)][0][q] != picks[("orig", i)][0][q]
                             for (oo, i) in picks if oo == o and ("orig", i) in picks for q in picks[(o, i)][1]]
                        flip[o] = pct(sum(f) / len(f)) if f else "–"
                n = sum(len(t) for (oo, _), (_, t) in picks.items() if oo == "orig")
                print(f"| {arm} | {src} | {k} | {n} | {acc['orig']} | {acc['again']} | {acc['rev']} | {acc['shuf']} | "
                      f"{flip['again']} | {flip['rev']} | {flip['shuf']} |")
    for m in meta:
        if m["calls"] and m["arm"] == "djev":
            cps = m["calls"] / m["seconds"]
            print(f"\ndjev pass {m['at']}: {m['calls']} calls in {m['seconds']} s at {m['workers']} parallel = "
                  f"{cps:.1f} calls/s, ${3.19 / 3600 / cps * 1000:.3f} per 1,000 calls while the GPU is busy.")


def main(argv):
    if argv[:1] == ["run"]:
        return cmd_run(argv[1:]) or 0
    if argv[:1] == ["report"]:
        return cmd_report(argv[1:]) or 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
