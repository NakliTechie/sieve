#!/usr/bin/env python3
"""Can zero-shot answers replace human labels? Train students on a teacher's labels, score them on gold test rows.

  .venv/bin/python pipeline/distill.py label <bench> --per-opt N    # GPU service: djev@3 reads of the training pool
  .venv/bin/python pipeline/distill.py run <bench> --per-opt N      # local: GLiNER reads, label sets, train, score
  .venv/bin/python pipeline/distill.py review <bench> --per-opt N --teacher djev@3|gliner|gliner-multi
                                                                  # review-budget sweep: correct 0/10/25/50 % (+ random 25 %)
  python3 pipeline/distill.py report <bench>

Pool: the first N training rows of each intent (curve.stage_rows; gold labels known, used only for the "gold",
"spot" and "noise" variants and to measure label error). Test: the bench's fixed 1,000 gold rows.
Label sets (each trains Ettin-17M and mmBERT-small):
  gold          the pool's human labels (the reference: what Batch C measured)
  djev3         djev averaged over 3 reads in shuffled option orders, argmax, every row
  djev3-top50   the half of the rows where djev3 is most confident; top25 the quarter
  agree         rows where djev3 and GLiNER (gliner-multi for Hindi) pick the same intent
  gliner        GLiNER's argmax, every row
  spot10/25     djev3 labels, with the 10 % / 25 % least-confident rows corrected to gold (a person checking those)
  noise10..40   gold with that share of labels flipped to a random other intent (how much label error training absorbs)
Reads are cached in data/bench/<bench>/cache/ (the same keys as curve.py). Results append to
data/bench/<bench>/distill.jsonl (resumable). Training runs under backends.gpu_lock.
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import curve as C  # noqa: E402
import djcore as P  # noqa: E402

MODELS = {"ettin": (B.TRAINED["tiny"][0], 1e-4), "mmbert": (B.TRAINED["mmbert"][0], 5e-5)}
MMBERT_VARIANTS = {"gold", "djev3", "djev3-top50", "agree", "spot25", "noise20"}  # mmBERT is 4-6x slower: key sets only


def pool(name, per_opt):
    d, train, test = C.load(name)
    return d, C.stage_rows(train, per_opt), test, sorted({l for _, l in train})


def label_sets(name, pool_rows, labels):
    d = P.DATA / "bench" / name
    texts, gold = [t for t, _ in pool_rows], [l for _, l in pool_rows]
    t3 = C.zero_shot("djev@3", d, labels, texts)
    g = C.zero_shot("gliner-multi" if name.endswith("-hi") else "gliner", d, labels, texts)
    top = lambda p: max(p, key=p.get)
    conf = [max(p.values()) for p in t3]
    order = sorted(range(len(texts)), key=lambda i: -conf[i])  # most confident first
    sets = {"gold": list(zip(texts, gold)), "djev3": [(t, top(p)) for t, p in zip(texts, t3)],
            "gliner": [(t, top(p)) for t, p in zip(texts, g)]}
    for share, nm in ((0.5, "djev3-top50"), (0.25, "djev3-top25")):
        keep = set(order[:int(len(order) * share)])
        sets[nm] = [(texts[i], top(t3[i])) for i in range(len(texts)) if i in keep]
    sets["agree"] = [(t, top(a)) for t, a, b in zip(texts, t3, g) if top(a) == top(b)]
    for share, nm in ((0.10, "spot10"), (0.25, "spot25")):
        fix = set(order[len(order) - int(len(order) * share):])  # least confident
        sets[nm] = [(texts[i], gold[i] if i in fix else top(t3[i])) for i in range(len(texts))]
    rng = random.Random(0)
    for share in (10, 20, 30, 40):
        flip = set(rng.sample(range(len(texts)), int(len(texts) * share / 100)))
        sets[f"noise{share}"] = [(texts[i], rng.choice([l for l in labels if l != gold[i]]) if i in flip else gold[i])
                                 for i in range(len(texts))]
    gmap = dict(zip(texts, gold))
    err = {k: round(sum(gmap[t] != l for t, l in v) / max(1, len(v)), 4) for k, v in sets.items()}
    return sets, err


def cmd_label(name, per_opt):
    _, rows, _, labels = pool(name, per_opt)
    t0 = time.time()
    C.zero_shot("djev@3", P.DATA / "bench" / name, labels, [t for t, _ in rows])
    print(f"verdict=LABELLED bench={name} rows={len(rows)} reads=3 seconds={time.time() - t0:.0f}")


def cmd_run(name, per_opt):
    import tiny as T
    d, rows, test, labels = pool(name, per_opt)
    out = d / "distill.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["per_opt"], r["set"], r["model"]))
            except ValueError:
                pass
    with B.gpu_lock():  # GLiNER reads use the GPU too: one GPU job per machine
        sets, err = label_sets(name, rows, labels)
    texts = [t for t, _ in test]
    for set_name, data in sets.items():
        for model in ("ettin", "mmbert"):
            if model == "mmbert" and set_name not in MMBERT_VARIANTS or (str(per_opt), set_name, model) in done:
                continue
            model_id, lr = MODELS[model]
            epochs = min(30, max(5, -(-300 // max(1, -(-len(data) // 32)))))
            with B.gpu_lock():
                m, tok, secs = T.train(data, labels, model_id, epochs, lr, 64)
                probs = T.proba(m, tok, texts, 64)
                del m, tok
                C.free()
            acc, ll = C.scores(probs, test)
            rec = {"bench": name, "per_opt": str(per_opt), "set": set_name, "model": model, "rows": len(data),
                   "label_error": err[set_name], "acc": acc, "ll": ll, "train_s": round(secs, 1),
                   "at": time.strftime("%Y%m%d-%H%M%S")}
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("verdict=SCORED " + " ".join(f"{k}={v}" for k, v in rec.items() if k != "at"), flush=True)
    print(f"verdict=DONE bench={name} next: python3 pipeline/distill.py report {name}")


def cmd_review(name, per_opt, teacher):
    """Review-budget sweep: the teacher labels the pool; a person corrects k % of rows (the least-confident, or a random
    k % as the control); mmBERT (and Ettin at 25 %) train on the result. Records go to distill.jsonl with set names
    review<k>-<least|random>-<teacher>."""
    import tiny as T
    d, rows, test, labels = pool(name, per_opt)
    out = d / "distill.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["per_opt"], r["set"], r["model"]))
            except ValueError:
                pass
    texts, gold = [t for t, _ in rows], [l for _, l in rows]
    with B.gpu_lock():
        probs = C.zero_shot(teacher, d, labels, texts)
    top = lambda p: max(p, key=p.get)
    teach = [top(p) for p in probs]
    least = sorted(range(len(texts)), key=lambda i: max(probs[i].values()))  # least confident first
    rnd = list(range(len(texts)))
    random.Random(0).shuffle(rnd)
    plan = [(k, "least") for k in (0, 10, 25, 50)] + [(25, "random")]
    for k, sel in plan:
        fix = set((least if sel == "least" else rnd)[:int(len(texts) * k / 100)])
        data = [(texts[i], gold[i] if i in fix else teach[i]) for i in range(len(texts))]
        err = round(sum(l != g for (_, l), g in zip(data, gold)) / len(data), 4)
        set_name = f"review{k}-{sel}-{teacher}"
        for model in ("mmbert", "ettin"):
            if model == "ettin" and (k, sel) != (25, "least") or (str(per_opt), set_name, model) in done:
                continue
            model_id, lr = MODELS[model]
            epochs = min(30, max(5, -(-300 // max(1, -(-len(data) // 32)))))
            with B.gpu_lock():
                m, tok, secs = T.train(data, labels, model_id, epochs, lr, 256 if name == "civil" and model == "ettin" else 64)
                pr = T.proba(m, tok, [t for t, _ in test], 64)
                del m, tok
                C.free()
            acc, ll = C.scores(pr, test)
            rec = {"bench": name, "per_opt": str(per_opt), "set": set_name, "model": model, "rows": len(data),
                   "reviewed": len(fix), "label_error": err, "acc": acc, "ll": ll, "train_s": round(secs, 1),
                   "teacher": teacher, "at": time.strftime("%Y%m%d-%H%M%S")}
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("verdict=SCORED " + " ".join(f"{k2}={v}" for k2, v in rec.items() if k2 != "at"), flush=True)
    print(f"verdict=DONE bench={name} review teacher={teacher}")


def cmd_report(name):
    d = P.DATA / "bench" / name
    recs = {}
    for line in (d / "distill.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs[(r["per_opt"], r["set"], r["model"])] = r
    stock = {}
    for line in (d / "curve.jsonl").read_text().splitlines():
        r = json.loads(line)
        if str(r["stage"]) == "0":
            stock[r["arm"]] = r["acc"]
    n = 1000
    print(f"Teacher on the test rows (stock, no labels): djev@3 {100 * stock.get('djev@3', float('nan')):.1f} %, "
          f"GLiNER {100 * stock.get('gliner', float('nan')):.1f} %. One standard error at 1,000 rows is about "
          f"{100 * math.sqrt(0.25 / n):.1f} pts.\n")
    print("| Pool (per intent) | Label set | Rows | Label error | Ettin-17M | mmBERT-small |\n|---|---|---|---|---|---|")
    order = ["gold", "djev3", "djev3-top50", "djev3-top25", "agree", "gliner", "spot10", "spot25",
             "noise10", "noise20", "noise30", "noise40"]
    for po in sorted({k[0] for k in recs}, key=lambda s: int(s) if s.isdigit() else 10 ** 9):
        for s in order:
            e, m = recs.get((po, s, "ettin")), recs.get((po, s, "mmbert"))
            if not (e or m):
                continue
            r = e or m
            cell = lambda x: f"{100 * x['acc']:.1f}" if x else "–"
            print(f"| {po} | {s} | {r['rows']} | {100 * r['label_error']:.1f} % | {cell(e)} | {cell(m)} |")


def main(argv):
    opt = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)}
    if len(argv) >= 2 and argv[0] == "review":
        return cmd_review(argv[1], opt.get("--per-opt", "100"), opt.get("--teacher", "gliner")) or 0
    if len(argv) >= 2 and argv[0] in ("label", "run"):
        per_opt = opt.get("--per-opt", "50")
        return (cmd_label if argv[0] == "label" else cmd_run)(argv[1], per_opt) or 0
    if len(argv) >= 2 and argv[0] == "report":
        return cmd_report(argv[1]) or 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
