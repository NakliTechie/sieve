#!/usr/bin/env python3
"""Stock vs trained at every stage: one fixed test set, label counts 0 / 5 / 10 / 20 / 50 / all per option.

  .venv/bin/python pipeline/curve.py run <bench> [--arms a,b,...] [--stages 5,10,20,50,all] [--cal-max 20] [--seed 0]
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
LOCAL = "gliner,gliner-multi,gliner+cal,gliner-multi+cal,tfidf,tiny,mmbert"
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


def stage_rows(train, stage, seed=0):
    """The first `stage` rows of each label after a seeded shuffle; stages are nested within a seed."""
    by = {}
    for t, l in train:
        by.setdefault(l, []).append((t, l))
    rng = random.Random(seed)
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


def tfidf_proba(rows, labels, texts):
    """TF-IDF (word 1-2 grams + character 2-5 grams) + logistic regression: the keyword baseline. CPU, seconds."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                     TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))
    clf = make_pipeline(vec, LogisticRegression(max_iter=2000, C=10))
    t0 = time.time()
    clf.fit([t for t, _ in rows], [l for _, l in rows])
    secs = time.time() - t0
    classes = list(clf.classes_)
    out = [{l: 0.0 for l in labels} | dict(zip(classes, p)) for p in clf.predict_proba(texts)]
    return out, secs


def never_predicted(probs, test):
    """Labels present in the test rows that the model never predicts: the collapse check."""
    picked = {max(p, key=p.get) for p in probs}
    return len({l for _, l in test} - picked)


def run(name, arms, stages, cal_max, seed=0):
    import djp
    d, train, test = load(name)
    labels = sorted({l for _, l in train})
    out = d / "curve.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["arm"], str(r["stage"]), r.get("seed", 0)))
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
            if (arm, "0", 0) in done or seed != 0:  # stock arms have no labels, so no seed
                continue
            t0 = time.time()
            acc, ll = scores(zero_shot(arm, d, labels, texts), test)
            record({"arm": arm, "stage": 0, "rows": 0, "acc": acc, "ll": ll, "seconds": round(time.time() - t0, 1)})
            continue
        for st in stages:
            if (arm, str(st), seed) in done:
                continue
            rows = stage_rows(train, st, seed)
            if arm.endswith("+cal"):
                if st == "all" or int(st) > cal_max:
                    continue
                t0 = time.time()
                tr_p = zero_shot(base, d, labels, [t for t, _ in rows])
                bias = djp.fit_bias(labels, tr_p, [l for _, l in rows])
                te_p = [P.calibrate(p, bias) for p in zero_shot(base, d, labels, texts)]
                acc, ll = scores(te_p, test)
                record({"arm": arm, "stage": st, "rows": len(rows), "acc": acc, "ll": ll,
                        "seconds": round(time.time() - t0, 1), "seed": seed})
            elif arm in ("tiny", "mmbert"):
                import tiny as T
                steps_per_epoch = max(1, -(-len(rows) // 32))
                epochs = min(30, max(5, -(-300 // steps_per_epoch)))
                lr = 1e-4 if arm == "tiny" else 5e-5
                fixed = arm == "mmbert"
                with B.gpu_lock():  # one model on the GPU per machine (backends.gpu_lock)
                    model, tok, secs = T.train(rows, labels, MODELS[arm], epochs, lr, maxlen, seed=seed, fixed_pad=fixed)
                    probs = T.proba(model, tok, texts, maxlen, 64 if fixed else 128, fixed_pad=fixed)
                    del model, tok
                    free()
                acc, ll = scores(probs, test)
                record({"arm": arm, "stage": st, "rows": len(rows), "acc": acc, "ll": ll, "seconds": round(secs, 1),
                        "epochs": epochs, "seed": seed, "never_predicted": never_predicted(probs, test)})
            elif arm == "tfidf":
                probs, secs = tfidf_proba(rows, labels, texts)
                acc, ll = scores(probs, test)
                record({"arm": arm, "stage": st, "rows": len(rows), "acc": acc, "ll": ll, "seconds": round(secs, 1),
                        "seed": seed, "never_predicted": never_predicted(probs, test)})
            else:
                raise SystemExit(f"verdict=SETUP unknown arm '{arm}'; arms: {LOCAL},djev,djev@3,djev+cal")
    print(f"verdict=DONE bench={name} next: python3 pipeline/curve.py report {name}")


def report(name):
    """Markdown table; cells with several seeds show mean ± standard deviation across seeds (n in the header)."""
    import statistics
    d = P.DATA / "bench" / name
    recs = {}
    for line in (d / "curve.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs.setdefault((r["arm"], str(r["stage"])), {})[r.get("seed", 0)] = r  # a re-run of a seed replaces it
    n = next(iter(next(iter(recs.values())).values()))["test"]
    stages = ["0"] + [s for s in ("5", "10", "20", "50", "all") if any(k[1] == s for k in recs)]
    order = ["gliner", "gliner-multi", "djev", "djev@3", "gliner+cal", "gliner-multi+cal", "djev+cal", "tfidf", "tiny",
             "mmbert"]
    arms = [a for a in order if any(k[0] == a for k in recs)]
    rows_at = {s: next((r["rows"] for (a, st), v in recs.items() for r in v.values() if st == s and r["rows"]), 0)
               for s in stages}
    mean = lambda v: statistics.mean(r["acc"] for r in v.values())
    seeds = max(len(v) for v in recs.values())
    print(f"Test rows: {n} (the same rows for every cell; one standard error is about "
          f"{100 * math.sqrt(0.25 / n):.1f} pts at 50 %). Cells: accuracy %"
          + (f", mean ± standard deviation over up to {seeds} seeds (each seed draws different labelled rows and "
             f"starts training differently)" if seeds > 1 else "") +
          "; train seconds in brackets; 'k unused' = k test intents the model never predicts. "
          "Stage = labelled rows per option (0 = stock, no labels).\n")
    print("| Arm | " + " | ".join(f"{s}/opt ({rows_at[s]} rows)" if s != "0" else "stock" for s in stages) + " |")
    print("|---" * (len(stages) + 1) + "|")
    for a in arms:
        cells = []
        for s in stages:
            v = recs.get((a, s))
            if not v:
                cells.append("–")
                continue
            accs = [100 * r["acc"] for r in v.values()]
            cell = f"{statistics.mean(accs):.1f}" + (f" ± {statistics.stdev(accs):.1f}" if len(accs) > 1 else "")
            if a in ("tiny", "mmbert", "tfidf"):
                cell += f" [{statistics.mean(r['seconds'] for r in v.values()):.0f} s]"
            unused = max(r.get("never_predicted", 0) for r in v.values())
            if unused:
                cell += f" ({unused} unused)"
            cells.append(cell)
        print(f"| {a} | " + " | ".join(cells) + " |")
    stock = [mean(v) for (a, s), v in recs.items() if a in STOCK]
    if stock:
        best = max(stock)
        se = math.sqrt(best * (1 - best) / n)
        cross = next((s for s in stages[1:] if any(a in ("tiny", "mmbert", "tfidf") and (a, s) in recs
                                                   and mean(recs[(a, s)]) > best + 2 * se for a in arms)), None)
        print(f"\nBest stock: {100 * best:.1f} %. A trained model first beats it by more than 2 standard errors at: "
              f"{cross + ' labels per option' if cross else 'no stage measured'}.")


def main(argv):
    if len(argv) >= 2 and argv[0] == "run":
        opt = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)}
        arms = opt.get("--arms", LOCAL).split(",")
        stages = opt.get("--stages", "5,10,20,50,all").split(",")
        return run(argv[1], arms, stages, int(opt.get("--cal-max", 20)), int(opt.get("--seed", 0))) or 0
    if len(argv) >= 2 and argv[0] == "report":
        return report(argv[1]) or 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
