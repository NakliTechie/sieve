#!/usr/bin/env python3
"""Reproduce the GLiNER2.5-Decide model card's headline on fastino/fast-decisions, the card's way.

  .venv/bin/python pipeline/gliner_card.py [--threshold 0.4] [--model fastino/GLiNER2.5-Decide]

The card (huggingface.co/fastino/GLiNER2.5-Decide, read 2026-09-26) reports 60.2 % exact-match accuracy on
"17 domains, 300 held-out examples each", with the same text and candidate labels for every model. Our A/B
(results/ab-2026-09-25.md) scored single-choice questions only and got 67.3 %. This script scores every question,
multi-label ones included (exact match = predicted label set equals the true set; multi-label questions use the
card's example cls_threshold, 0.4), and prints single-choice, multi-label and all, overall and per domain.
Rows: data/bench/fast-decisions/*.jsonl (the published files: 100 rows per domain). Local, free. Results append to
data/ab/gliner-card/results.jsonl; the verdict line carries the totals.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import djcore as P  # noqa: E402


def main(argv):
    opt = {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}
    thr = float(opt.get("--threshold", 0.4))
    model_id = opt.get("--model", B.GLINER_MODEL)
    B.GLINER_MODEL = model_id
    m = B._gliner()
    out = P.DATA / "ab" / "gliner-card"
    out.mkdir(parents=True, exist_ok=True)
    tally = {}  # (domain, kind) -> [hits, n]
    t0 = time.time()
    with open(out / "results.jsonl", "a") as f:
        for path in sorted((P.DATA / "bench" / "fast-decisions").glob("*.jsonl")):
            for i, line in enumerate(path.read_text().splitlines()):
                r = json.loads(line)
                schema = m.create_schema()
                cls = r["output"]["classifications"]
                for c in cls:
                    if c["multi_label"]:
                        schema = schema.classification(c["task"], list(c["labels"]), multi_label=True, cls_threshold=thr)
                    else:
                        schema = schema.classification(c["task"], list(c["labels"]))
                res = m.extract(r["input"], schema, include_confidence=True)
                for c in cls:
                    got = res.get(c["task"])
                    items = got if isinstance(got, list) else [got] if got else []
                    labels = sorted({(x.get("label") if isinstance(x, dict) else x[0] if isinstance(x, (list, tuple)) else x)
                                     for x in items})
                    hit = labels == sorted(c["true_label"])
                    kind = "multi" if c["multi_label"] else "single"
                    for key in ((path.stem, kind), ("_all", kind)):
                        t = tally.setdefault(key, [0, 0])
                        t[0] += hit
                        t[1] += 1
                    f.write(json.dumps({"model": model_id, "threshold": thr, "domain": path.stem, "row": i,
                                        "task": c["task"], "kind": kind, "pred": labels,
                                        "truth": sorted(c["true_label"]), "hit": hit}) + "\n")
    pct = lambda k: f"{100 * tally[k][0] / tally[k][1]:.1f}% ({tally[k][1]})" if k in tally else "–"
    print("| Domain | Single-choice | Multi-label | All |\n|---|---|---|---|")
    for d in sorted({k[0] for k in tally}):
        both = [tally.get((d, k), [0, 0]) for k in ("single", "multi")]
        allp = f"{100 * sum(b[0] for b in both) / sum(b[1] for b in both):.1f}%"
        print(f"| {d} | {pct((d, 'single'))} | {pct((d, 'multi'))} | {allp} |")
    s, mm = tally.get(("_all", "single"), [0, 0]), tally.get(("_all", "multi"), [0, 0])
    print(f"verdict=SCORED model={model_id} threshold={thr} single={s[0]}/{s[1]} multi={mm[0]}/{mm[1]} "
          f"all={(s[0] + mm[0]) / max(1, s[1] + mm[1]):.4f} seconds={time.time() - t0:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
