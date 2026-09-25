#!/usr/bin/env python3
"""Fine-tune a tiny encoder classifier in seconds, per question; the "train on the fly" arm.

  .venv/bin/python pipeline/tiny.py banking77 [--per-class N] [--model M] [--epochs E]
  .venv/bin/python pipeline/tiny.py customer <id> [--model M] [--epochs E]

banking77: train on data/bench/banking77/train.csv (optionally N rows per class), score the 3,080-message test set.
customer: train on the customer's training split (djcore.split), score their held-out rows, one classifier per
question: the same rows pipeline/ab.py scored for djev and GLiNER.

Default model jhu-clsp/ettin-encoder-17m (MIT). Runs on Apple MPS, CUDA or CPU. Prints one verdict line with accuracy,
train seconds and rows; appends it to data/tiny/log.jsonl. Needs torch + transformers (the .venv env).
"""
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djcore as P  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
LOG = P.DATA / "tiny" / "log.jsonl"


def train_eval(train, test, labels, model_id, epochs, lr, max_len, bs=32, seed=0):
    """train/test: [(text, label)]. -> (accuracy, train_seconds, predictions)."""
    torch.manual_seed(seed)
    random.seed(seed)
    idx = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=idx).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * -(-len(train) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))
    enc = lambda texts: tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEV)
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        random.shuffle(train)
        for i in range(0, len(train), bs):
            b = train[i:i + bs]
            out = model(**enc([t for t, _ in b]), labels=torch.tensor([idx[l] for _, l in b], device=DEV))
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
    if DEV == "mps":
        torch.mps.synchronize()
    secs = time.time() - t0
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test), 128):
            logits = model(**enc([t for t, _ in test[i:i + 128]])).logits
            preds += [labels[j] for j in logits.argmax(-1).tolist()]
    acc = sum(p == l for p, (_, l) in zip(preds, test)) / len(test)
    return acc, secs, preds


def log(rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("verdict=TRAINED " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)


def cmd_banking77(opt):
    d = P.DATA / "bench" / "banking77"
    read = lambda f: [(r["text"], r["category"]) for r in csv.DictReader(open(d / f, encoding="utf-8"))]
    train, test = read("train.csv"), read("test.csv")
    labels = sorted({l for _, l in train})
    n = int(opt.get("--per-class", 0))
    if n:
        rng, by = random.Random(0), {}
        for t, l in train:
            by.setdefault(l, []).append((t, l))
        train = [x for l in labels for x in rng.sample(by[l], min(n, len(by[l])))]
    model = opt.get("--model", "jhu-clsp/ettin-encoder-17m")
    acc, secs, _ = train_eval(train, test, labels, model, int(opt.get("--epochs", 5)), float(opt.get("--lr", 1e-4)), 64)
    log({"task": "banking77", "model": model, "device": DEV, "train_rows": len(train), "per_class": n or "all",
         "test_rows": len(test), "accuracy": round(acc, 4), "train_s": round(secs, 1)})


def cmd_customer(cid, opt):
    customer, rows = P.load_customer(cid)
    train, hold = P.split(rows)
    model = opt.get("--model", "jhu-clsp/ettin-encoder-17m")
    total_s, hits, n = 0.0, 0, 0
    for q, spec in customer["questions"].items():
        labels = list(spec["criteria"])
        acc, secs, _ = train_eval([(r["state"], r["answers"][q]) for r in train], [(r["state"], r["answers"][q]) for r in hold],
                                  labels, model, int(opt.get("--epochs", 10)), float(opt.get("--lr", 1e-4)), 256)
        total_s += secs
        hits += round(acc * len(hold))
        n += len(hold)
        log({"task": f"customer:{cid}:{q}", "model": model, "device": DEV, "train_rows": len(train),
             "test_rows": len(hold), "accuracy": round(acc, 4), "train_s": round(secs, 1)})
    print(f"customer={cid} all-questions accuracy={hits / n:.4f} train_s={total_s:.1f}")


def main(argv):
    if not argv or argv[0] not in ("banking77", "customer"):
        print(__doc__)
        return 2
    if argv[0] == "banking77":
        cmd_banking77({argv[i]: argv[i + 1] for i in range(1, len(argv) - 1, 2)})
    else:
        cmd_customer(argv[1], {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
