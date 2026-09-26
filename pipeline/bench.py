#!/usr/bin/env python3
"""Real, human-labelled benchmark sets for stock-vs-trained curves (pipeline/curve.py).

  .venv/bin/python pipeline/bench.py list
  .venv/bin/python pipeline/bench.py fetch <name>|all     # -> data/bench/<name>/{train,test}.csv (text,label)

Every set is written the same way: train.csv (the training pool) and test.csv (a seeded sample of at most 1,000
rows, the same rows for every arm, stock and trained). Parquet comes from the Hugging Face datasets-server
(no account needed for these; all ungated). Needs pyarrow (the .venv).
"""
import csv
import io
import json
import random
import sys
import urllib.request

import pyarrow.parquet as pq

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djcore as P  # noqa: E402

API = "https://huggingface.co/api/datasets"
TEST_ROWS = 1000

RECIPES = {
    "massive-en": {"dataset": "SetFit/amazon_massive_intent_en-US", "license": "CC-BY-4.0 (AmazonScience/massive)",
                   "what": "real voice-assistant utterances, 60 intents, English"},
    "massive-hi": {"dataset": "SetFit/amazon_massive_intent_hi-IN", "license": "CC-BY-4.0 (AmazonScience/massive)",
                   "what": "the same 60 intents, Hindi (Devanagari)"},
    "clinc": {"dataset": "clinc/clinc_oos", "config": "plus", "license": "CC-BY-3.0",
              "what": "crowdsourced assistant queries, 150 intents + out-of-scope ('oos')"},
    "civil": {"dataset": "google/civil_comments", "license": "CC0-1.0",
              "what": "real news comments; toxic = toxicity >= 0.5; train pool and test balanced 50/50"},
    "hinglish-top": {"dataset": "WillHeld/hinglish_top", "license": "Apache-2.0 (google-research-datasets/Hinglish-TOP)",
                     "what": "human-written code-switched Hinglish assistant queries, top-level intent (arXiv 2211.07514)"},
    "hinglish-top-en": {"dataset": "WillHeld/hinglish_top", "license": "Apache-2.0",
                        "what": "the same queries' English originals and intents: the English twin of hinglish-top"},
    "hinglish-yt": {"dataset": "shae2977/hinglish-youtube-sentiments-dataset", "license": "CC-BY-4.0",
                    "what": "3,190 real Hinglish YouTube comments, hand-labelled sentiment (3 classes)"},
    "hinglish-retail": {"dataset": "Hari5115/hinglish-retail-intent-dataset", "license": "MIT",
                        "what": "SYNTHETIC Hinglish e-commerce support messages, intents (messaging-platform-shaped; not real traffic)"},
    "banking77": {"local": True, "license": "CC-BY-4.0 (PolyAI/banking77)", "what": "real bank queries, 77 intents"},
}


def parquet(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return pq.read_table(io.BytesIO(r.read())).to_pylist()


def split_urls(dataset, config="default"):
    with urllib.request.urlopen(f"{API}/{dataset}/parquet", timeout=60) as r:
        return json.loads(r.read())[config]


def label_names(dataset, config, feature):
    url = f"https://datasets-server.huggingface.co/info?dataset={dataset}&config={config}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())["dataset_info"]["features"][feature]["names"]


def write(name, train, test):
    d = P.DATA / "bench" / name
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    if len(test) > TEST_ROWS:
        test = rng.sample(test, TEST_ROWS)
    for fname, rows in (("train.csv", train), ("test.csv", test)):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["text", "label"])
        w.writerows((t.replace("\n", " ").strip(), l) for t, l in rows if t and t.strip())
        P.write_atomic(d / fname, buf.getvalue())
    labels = sorted({l for _, l in train})
    print(f"verdict=FETCHED bench={name} train={len(train)} test={len(test)} labels={len(labels)} dir={d}")


def fetch(name):
    rec = RECIPES[name]
    if rec.get("local"):
        d = P.DATA / "bench" / name
        rows = lambda f: [(r["text"], r["category"]) for r in csv.DictReader(open(d / f, encoding="utf-8"))]
        if not (d / "train.csv").exists():
            raise SystemExit(f"verdict=SETUP {d}/train.csv missing (banking77 comes from the Batch A bench)")
        if "category" in open(d / "train.csv").readline():  # the original files: keep them, add the curve copies
            train, test = rows("train.csv"), rows("test.csv")
            name = "banking77-curve"
            write(name, train, test)
        return
    urls = split_urls(rec["dataset"], rec.get("config", "default"))
    if name.startswith("massive"):
        get = lambda split: [(r["text"], r["label_text"]) for u in urls[split] for r in parquet(u)]
        write(name, get("train"), get("test"))
    elif name == "clinc":
        names = label_names(rec["dataset"], "plus", "intent")
        get = lambda split: [(r["text"], names[r["intent"]]) for u in urls[split] for r in parquet(u)]
        write(name, get("train"), get("test"))
    elif name.startswith("hinglish-top"):
        import re
        col = "en_query" if name.endswith("-en") else "cs_query"
        intent = lambda r: re.match(r"\[IN:(\w+)", r["cs_parse"]).group(1).lower()
        get = lambda split: [(r[col], intent(r)) for u in urls[split] for r in parquet(u)]
        train = get("train") + get("validation")  # 2,993 + 1,390: the published train split alone is small
        keep = {l for l, n in __import__("collections").Counter(l for _, l in train).items() if n >= 5}
        write(name, [x for x in train if x[1] in keep], [x for x in get("test") if x[1] in keep])
    elif name == "hinglish-yt":
        rows = [(r["comment"], r["sentiment"].lower()) for u in urls["train"] for r in parquet(u)]
        rng = random.Random(0)
        rng.shuffle(rows)
        write(name, rows[:len(rows) - 800], rows[len(rows) - 800:])  # single published split: hold out 800
    elif name == "hinglish-retail":
        get = lambda split: [(r["text"], r["label"]) for u in urls[split] for r in parquet(u)]
        write(name, get("train") + get("validation"), get("test"))
    elif name == "civil":
        lab = lambda r: "toxic" if (r["toxicity"] or 0) >= 0.5 else "not_toxic"
        rng = random.Random(0)

        def balanced(rows, per):
            by = {}
            for r in rows:
                by.setdefault(lab(r), []).append((r["text"], lab(r)))
            return [x for k in sorted(by) for x in rng.sample(by[k], min(per, len(by[k])))]
        train = balanced([r for u in urls["validation"] for r in parquet(u)], 2000)  # validation split = train pool
        test = balanced([r for u in urls["test"] for r in parquet(u)], TEST_ROWS // 2)
        write(name, train, test)


def main(argv):
    if argv[:1] == ["list"] or not argv:
        for n, r in RECIPES.items():
            print(f"{n:11} {r.get('dataset', 'local'):40} {r['license']:34} {r['what']}")
        return 0
    if argv[0] == "fetch":
        for n in (RECIPES if argv[1:] == ["all"] else argv[1:]):
            fetch(n)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
