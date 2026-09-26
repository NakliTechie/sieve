#!/usr/bin/env python3
"""Stock vs trained at every stage: one fixed test set, label counts 0 / 5 / 10 / 20 / 50 / all per option.

  .venv/bin/python pipeline/curve.py run <bench> [--arms a,b,...] [--stages 5,10,20,50,all] [--cal-max 20]
  python3 pipeline/curve.py report <bench>             # markdown table -> stdout

Bench sets come from pipeline/bench.py (data/bench/<name>/{train,test}.csv). Stage N takes the first N rows of each
label from the training pool after one seeded shuffle, so stages are nested (the 10-row set contains the 5-row set).

Arms. Stock (no labels; stage 0): gliner, gliner-multi, djev, djev@3 (3 reads over shuffled option orders).
Using labels (every stage): gliner+cal, gliner-multi+cal, djev+cal (a per-option log-prior shift fitted on that
stage's rows; only up to --cal-max per option, since it needs a model read of every training row), tiny (Ettin-17M
fine-tuned on that stage's rows), mmbert (mmBERT-small, 140M, multilingual, fine-tuned the same way).
Metrics per (arm, stage): accuracy and log-loss on the test rows, rows used, train seconds. Records append to
data/bench/<name>/curve.jsonl; a re-run skips what is there (crash-safe, resumable). gliner and djev reads are cached
in data/bench/<name>/cache/. djev needs the proxy (DJEV_URL); it is never in the default --arms.
"""
import csv
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import djcore as P  # noqa: E402

URL = os.environ.get("DJEV_URL", "http://localhost:8081")
MODELS = {"gliner": "fastino/GLiNER2.5-Decide", "gliner-multi": "fastino/gliner2.5-multi-v1",
          "tiny": "jhu-clsp/ettin-encoder-17m", "mmbert": "jhu-clsp/mmBERT-small"}
LOCAL = "gliner,gliner-multi,gliner+cal,gliner-multi+cal,tiny,mmbert"
STOCK = ("gliner", "gliner-multi", "djev", "djev@3")


def free():
    """Return a trained model's memory before the next stage: without it, MPS keeps every stage's weights and optimiser
    state, and mmBERT-small (256k-token vocabulary) grew one run to 27 GB on a 24 GB Mac."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load(name):
    d = P.DATA / "bench" / name
    read = lambda f: [(r["text"], r["label"]) for r in csv.DictReader(open(d / f, encoding="utf-8"))]
    return d, read("train.csv"), read("test.csv")


def stage_rows(train, stage):
    by = {}
    for t, l in train:
        by.setdefault(l, []).append((t, l))
    rng = random.Random(0)
    for l in sorted(by):
        rng.shuffle(by[l])
    return [x for l in sorted(by) for x in (by[l] if stage == "all" else by[l][:int(stage)])]


def scores(probs, test):
    hits = sum(max(p, key=p.get) == l for p, (_, l) in zip(probs, test))
    ll = -sum(math.log(max(p.get(l, 0.0), 1e-6)) for p, (_, l) in zip(probs, test)) / len(test)
    return round(hits / len(test), 4), round(ll, 4)


def zero_shot(arm, d, labels, texts):
    """-> [{label: p}] for texts from a stock model, cached per bench."""
    qs = {"label": {"type": "choice", "instructions": "Which category fits this text?", "criteria": labels}}
    if arm.startswith("gliner"):
        mid = MODELS[arm]
        if B.GLINER_MODEL != mid:
            B.GLINER_MODEL, B._gl = mid, None
        return [p["label"] for p in B.gliner_read(qs, texts, B.Cache(d / "cache" / f"{arm}.jsonl"))]
    reads = 3 if arm == "djev@3" else 1
    return [p["label"] for p in B.djev_read(URL, qs, None, texts, B.Cache(d / "cache" / "djev.jsonl"), 8, reads)]


def run(name, arms, stages, cal_max):
    import djp
    d, train, test = load(name)
    labels = sorted({l for _, l in train})
    out = d / "curve.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["arm"], str(r["stage"])))
            except ValueError:
                pass
    texts = [t for t, _ in test]
    maxlen = 256 if name == "civil" else 64

    def record(rec):
        with open(out, "a") as f:
            f.write(json.dumps({**rec, "bench": name, "test": len(test), "at": time.strftime("%Y%m%d-%H%M%S")}) + "\n")
        print("verdict=SCORED " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)

    for arm in arms:
        base = arm.split("+")[0]
        if arm in STOCK:
            if (arm, "0") in done:
                continue
            t0 = time.time()
            acc, ll = scores(zero_shot(arm, d, labels, texts), test)
            record({"arm": arm, "stage": 0, "rows": 0, "acc": acc, "ll": ll, "seconds": round(time.time() - t0, 1)})
            continue
        for st in stages:
            if (arm, str(st)) in done:
                continue
            rows = stage_rows(train, st)
            if arm.endswith("+cal"):
                if st == "all" or int(st) > cal_max:
                    continue
                t0 = time.time()
                tr_p = zero_shot(base, d, labels, [t for t, _ in rows])
                bias = djp.fit_bias(labels, tr_p, [l for _, l in rows])
                te_p = [P.calibrate(p, bias) for p in zero_shot(base, d, labels, texts)]
                acc, ll = scores(te_p, test)
                record({"arm": arm, "stage": st, "rows": len(rows), "acc": acc, "ll": ll,
                        "seconds": round(time.time() - t0, 1)})
            elif arm in ("tiny", "mmbert"):
                import tiny as T
                steps_per_epoch = max(1, -(-len(rows) // 32))
                epochs = min(30, max(5, -(-300 // steps_per_epoch)))
                lr = 1e-4 if arm == "tiny" else 5e-5
                fixed = arm == "mmbert"
                import fcntl
                with open(P.DATA / ".train.lock", "w") as lock:  # one training per machine (see backends.tiny_fit)
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    model, tok, secs = T.train(rows, labels, MODELS[arm], epochs, lr, maxlen, fixed_pad=fixed)
                acc, ll = scores(T.proba(model, tok, texts, maxlen, 64 if fixed else 128, fixed_pad=fixed), test)
                record({"arm": arm, "stage": st, "rows": len(rows), "acc": acc, "ll": ll, "seconds": round(secs, 1),
                        "epochs": epochs})
                del model, tok
                free()
            else:
                raise SystemExit(f"verdict=SETUP unknown arm '{arm}'; arms: {LOCAL},djev,djev@3,djev+cal")
    print(f"verdict=DONE bench={name} next: python3 pipeline/curve.py report {name}")


def report(name):
    d = P.DATA / "bench" / name
    recs = {}
    for line in (d / "curve.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs[(r["arm"], str(r["stage"]))] = r
    n = next(iter(recs.values()))["test"]
    stages = ["0"] + [s for s in ("5", "10", "20", "50", "all") if any(k[1] == s for k in recs)]
    order = ["gliner", "gliner-multi", "djev", "djev@3", "gliner+cal", "gliner-multi+cal", "djev+cal", "tiny", "mmbert"]
    arms = [a for a in order if any(k[0] == a for k in recs)]
    rows_at = {s: next((r["rows"] for (a, st), r in recs.items() if st == s and r["rows"]), 0) for s in stages}
    print(f"Test rows: {n} (the same rows for every cell; one standard error is about "
          f"{100 * math.sqrt(0.25 / n):.1f} pts at 50 %). Cells: accuracy % / log-loss; train seconds in brackets for "
          f"trained arms. Stage = labelled rows per option (0 = stock, no labels).\n")
    print("| Arm | " + " | ".join(f"{s}/opt ({rows_at[s]} rows)" if s != "0" else "stock" for s in stages) + " |")
    print("|---" * (len(stages) + 1) + "|")
    for a in arms:
        cells = []
        for s in stages:
            r = recs.get((a, s))
            if not r:
                cells.append("–")
                continue
            extra = f" [{r['seconds']:.0f} s]" if a in ("tiny", "mmbert") else ""
            cells.append(f"{100 * r['acc']:.1f} / {r['ll']:.2f}{extra}")
        print(f"| {a} | " + " | ".join(cells) + " |")
    best_stock = max((r["acc"] for (a, s), r in recs.items() if a in STOCK), default=None)
    if best_stock is not None:
        se = math.sqrt(best_stock * (1 - best_stock) / n)
        cross = next((s for s in stages[1:] if any(recs.get((a, s), {}).get("acc", 0) > best_stock + 2 * se
                                                    for a in ("tiny", "mmbert"))), None)
        print(f"\nBest stock: {100 * best_stock:.1f} %. A trained model first beats it by more than 2 standard errors at: "
              f"{cross + ' labels per option' if cross else 'no stage measured'}.")


def main(argv):
    if len(argv) >= 2 and argv[0] == "run":
        opt = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)}
        arms = opt.get("--arms", LOCAL).split(",")
        stages = opt.get("--stages", "5,10,20,50,all").split(",")
        return run(argv[1], arms, stages, int(opt.get("--cal-max", 20))) or 0
    if len(argv) >= 2 and argv[0] == "report":
        return report(argv[1]) or 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
