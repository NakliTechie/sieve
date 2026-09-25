#!/usr/bin/env python3
"""A/B: djev (/v1/systemone) vs GLiNER2.5-Decide on the same rows, questions and options, plus an option-order test.

  .venv/bin/python pipeline/ab.py run [--arms djev,gliner] [--orders orig,rev,shuf] [--limit N]
  python3 pipeline/ab.py report [run-dir]          # aggregates -> stdout (markdown); raw rows stay in ~/.djev/ab

Suite: the held-out rows of every customer in $DJP_HOME and examples (djcore.split), plus the single-choice questions of
fastino/fast-decisions (~/.djev/bench/fast-decisions/*.jsonl, Apache-2.0). Both arms get the same information:
question name + options. No customer rules, no few-shot, no calibration: this compares base models.

Order sub-test: each row runs with options in the original order, the original order again (the noise floor: djev
randomises its answer slots per call), reversed, and shuffled (seeded per row).
Reported: accuracy per order, flip rate (answer changes when only the order changes), and position bias (how often
the first / last option shown is picked, against how often it is the right answer).

Results append to ~/.djev/ab/<run>/results.jsonl as they arrive; re-running `run` with the same --run skips what is done.
Env: DJEV_URL (default http://localhost:8081), DJEV_WORKERS (default 8), GLINER_MODEL (default fastino/GLiNER2.5-Decide).
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
import djcore as P  # noqa: E402

URL = os.environ.get("DJEV_URL", "http://localhost:8081")
WORKERS = int(os.environ.get("DJEV_WORKERS", "8"))
GLINER = os.environ.get("GLINER_MODEL", "fastino/GLiNER2.5-Decide")
BENCH = Path.home() / ".djev" / "bench" / "fast-decisions"
AB = Path.home() / ".djev" / "ab"


def suite(limit=None):
    """-> items {id, source, domain, text, questions {q: [options]}, truth {q: label}}."""
    items = []
    for d in P.all_customers():
        customer, rows = P.load_customer(d.name)
        if not rows:
            continue
        _, hold = P.split(rows)
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
    t0 = time.time()
    r = P.ask(URL, {q: {"type": "choice", "instructions": q.replace("_", " "), "criteria": o} for q, o in qs.items()}, text)
    return {q: r["answers"][q]["choice"] for q in qs}, \
           {q: r["answers"][q].get("confidence") for q in qs}, round((time.time() - t0) * 1000, 1), \
           r.get("diagnostics", {}).get("timing", {}).get("total_ms")


_gl, _gl_lock = None, threading.Lock()


def arm_gliner(text, qs):
    global _gl
    with _gl_lock:  # one model instance; GPU/MPS calls serialised
        if _gl is None:
            from gliner2 import AutoExtractor
            _gl = AutoExtractor.from_pretrained(GLINER)
        schema = _gl.create_schema()
        for q, o in qs.items():
            schema = schema.classification(q, o)
        t0 = time.time()
        r = _gl.extract(text, schema, include_confidence=True)
        ms = round((time.time() - t0) * 1000, 1)
    pick = {q: (r.get(q) or {}).get("label") if isinstance(r.get(q), dict) else r.get(q) for q in qs}
    conf = {q: (r.get(q) or {}).get("confidence") if isinstance(r.get(q), dict) else None for q in qs}
    return pick, conf, ms, ms


ARMS = {"djev": (arm_djev, WORKERS), "gliner": (arm_gliner, 1)}


def cmd_run(args):
    opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
    arms = opt.get("--arms", "djev,gliner").split(",")
    orders = opt.get("--orders", "orig,again,rev,shuf").split(",")
    limit = int(opt["--limit"]) if "--limit" in opt else None
    run = AB / opt.get("--run", time.strftime("%Y%m%d"))
    run.mkdir(parents=True, exist_ok=True)
    out = run / "results.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            x = json.loads(line)
            done.add((x["arm"], x["order"], x["id"]))
    items = suite(limit)
    lock = threading.Lock()
    for arm in arms:
        fn, workers = ARMS[arm]
        todo = [(o, it) for o in orders for it in items if (arm, o, it["id"]) not in done]
        print(f"arm={arm} items={len(items)} orders={orders} todo={len(todo)}", flush=True)
        t0, n = time.time(), [0]

        def one(job):
            order, it = job
            qs = ordered(it, order)
            try:
                pick, conf, ms, server_ms = fn(it["text"], qs)
                rec = {"arm": arm, "order": order, "id": it["id"], "source": it["source"], "domain": it["domain"],
                       "shown": qs, "truth": it["truth"], "pick": pick, "conf": conf, "ms": ms, "server_ms": server_ms}
            except Exception as e:  # recorded, not fatal: one bad row must not lose a 10-minute run
                rec = {"arm": arm, "order": order, "id": it["id"], "source": it["source"], "domain": it["domain"],
                       "error": f"{type(e).__name__}: {str(e)[:200]}"}
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n[0] += 1
                if n[0] % 200 == 0:
                    print(f"  {arm}: {n[0]}/{len(todo)} in {round(time.time() - t0)} s", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        print(f"arm={arm} finished {len(todo)} in {round(time.time() - t0)} s", flush=True)
    print(f"verdict=DONE run={run} next: python3 pipeline/ab.py report {run}")


def pct(x):
    return f"{100 * x:.1f}%"


def cmd_report(args):
    run = Path(args[0]) if args else sorted(AB.iterdir())[-1]
    recs = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines()]
    ok = [r for r in recs if "error" not in r]
    errs = [r for r in recs if "error" in r]
    arms = sorted({r["arm"] for r in ok})
    orders = [o for o in ("orig", "again", "rev", "shuf") if any(r["order"] == o for r in ok)]
    print(f"# A/B run {run.name}\n\n{len(ok)} scored calls, {len(errs)} errors. Accuracy is exact match per question.\n")
    # accuracy by source x arm x order
    print("## Accuracy\n\n| Rows | Questions | " + " | ".join(f"{a} {o}" for a in arms for o in orders) + " |")
    print("|---|---|" + "---|" * (len(arms) * len(orders)))
    groups = [("fast-decisions (all 16 domains)", lambda r: r["source"] == "fast-decisions")] + \
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
    if errs:
        print("\n## Errors\n")
        for e in sorted({(r['arm'], r['domain'], r['error']) for r in errs})[:10]:
            print(f"- {e[0]} {e[1]}: {e[2]}")


def main(argv):
    if argv[:1] == ["run"]:
        return cmd_run(argv[1:]) or 0
    if argv[:1] == ["report"]:
        return cmd_report(argv[1:]) or 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
