#!/usr/bin/env python3
"""Chain or single model for a two-language customer? English -> Ettin-17M, everything else -> mmBERT-small, against
one mmBERT (or one Ettin) trained on both languages.

  .venv/bin/python pipeline/chain.py [--stage 50|all]      # MASSIVE en-US + hi-IN (pipeline/bench.py fetch massive-en massive-hi)

Router: script only. A text goes to "other" when at least 30 % of its letters are Devanagari. It cannot see romanized
Hindi (Hinglish), which it sends to English; MASSIVE has none, so this run cannot measure that failure.
Measured: router accuracy against the known language; accuracy per language and overall on the 2 x 1,000 fixed test
rows; training seconds; serving latency per call (batch of 1) on CPU and on MPS. Appends one JSON line per run to
data/bench/chain.jsonl and prints a markdown table. Local, free; trains under backends.gpu_lock (one model per machine).
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import curve as C  # noqa: E402
import djcore as P  # noqa: E402


def is_other(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters) >= 0.3


def latency(model, tok, texts, max_len, device):
    """Median ms per single-text call on `device` (after 5 warm-up calls)."""
    import tiny as T
    model = model.to(device)
    for t in texts[:5]:
        T.proba(model, tok, [t], max_len)
    ms = []
    for t in texts[:200]:
        t0 = time.perf_counter()
        T.proba(model, tok, [t], max_len)
        if device == "mps":
            import torch
            torch.mps.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ms), 1)


def main(argv):
    import tiny as T
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "all"
    _, tr_en, te_en = C.load("massive-en")
    _, tr_hi, te_hi = C.load("massive-hi")
    labels = sorted({l for _, l in tr_en} | {l for _, l in tr_hi})
    en, hi = C.stage_rows(tr_en, stage), C.stage_rows(tr_hi, stage)
    router_ok = {"en": sum(not is_other(t) for t, _ in te_en) / len(te_en),
                 "hi": sum(is_other(t) for t, _ in te_hi) / len(te_hi)}
    specs = {"ettin": (B.TRAINED["tiny"][0], 1e-4, 64), "mmbert": (B.TRAINED["mmbert"][0], 5e-5, 64)}

    def fit(arm, rows):
        model_id, lr, max_len = specs[arm]
        epochs = min(30, max(5, -(-300 // max(1, -(-len(rows) // 32)))))
        with B.gpu_lock():
            m, tok, secs = T.train(rows, labels, model_id, epochs, lr, max_len)
            probs = {"en": T.proba(m, tok, [t for t, _ in te_en], max_len),
                     "hi": T.proba(m, tok, [t for t, _ in te_hi], max_len)}
            lat = {"mps": latency(m, tok, [t for t, _ in te_en + te_hi], max_len, "mps")}
            m = B.park(m)
        lat["cpu"] = latency(m, tok, [t for t, _ in te_en + te_hi], max_len, "cpu")
        del m
        C.free()
        return probs, round(secs, 1), lat

    acc = lambda probs, test: sum(max(p, key=p.get) == l for p, (_, l) in zip(probs, test)) / len(test)
    out = {"stage": stage, "train_rows": {"en": len(en), "hi": len(hi)}, "router_accuracy": router_ok, "arms": {}}
    for arm in ("ettin", "mmbert"):
        for name, rows in ((f"{arm}-en", en), (f"{arm}-hi", hi), (f"{arm}-both", en + hi)):
            probs, secs, lat = fit(arm, rows)
            out["arms"][name] = {"en": round(acc(probs["en"], te_en), 4), "hi": round(acc(probs["hi"], te_hi), 4),
                                 "train_s": secs, "latency_ms": lat, "_probs": probs}
            print(f"verdict=SCORED {name} en={out['arms'][name]['en']} hi={out['arms'][name]['hi']} "
                  f"train_s={secs} latency_ms={lat}", flush=True)

    def chain(en_arm, other_arm):
        """Route each test row by script; score with the model the router picked."""
        hits = {}
        for lang, test in (("en", te_en), ("hi", te_hi)):
            h = 0
            for j, (t, l) in enumerate(test):
                arm = other_arm if is_other(t) else en_arm
                p = out["arms"][arm]["_probs"][lang][j]
                h += max(p, key=p.get) == l
            hits[lang] = round(h / len(test), 4)
        return hits
    out["chains"] = {"ettin-en | mmbert-both": chain("ettin-en", "mmbert-both"),
                     "ettin-en | mmbert-hi": chain("ettin-en", "mmbert-hi"),
                     "mmbert-en | mmbert-hi": chain("mmbert-en", "mmbert-hi")}
    for a in out["arms"].values():
        a.pop("_probs")
    with open(P.DATA / "bench" / "chain.jsonl", "a") as f:
        f.write(json.dumps({**out, "at": time.strftime("%Y%m%d-%H%M%S")}) + "\n")
    both = lambda r: round((r["en"] + r["hi"]) / 2, 4)
    print(f"\nRouter (script) accuracy: English {100 * router_ok['en']:.1f} %, Hindi {100 * router_ok['hi']:.1f} %. "
          f"Stage: {stage} labels per option.\n")
    print("| Setup | English | Hindi | Both | Train s | ms/call CPU | ms/call MPS |\n|---|---|---|---|---|---|---|")
    for name, a in out["arms"].items():
        print(f"| {name} | {100 * a['en']:.1f} | {100 * a['hi']:.1f} | {100 * both(a):.1f} | {a['train_s']} | "
              f"{a['latency_ms']['cpu']} | {a['latency_ms']['mps']} |")
    for name, r in out["chains"].items():
        print(f"| chain: {name} | {100 * r['en']:.1f} | {100 * r['hi']:.1f} | {100 * both(r):.1f} | – | – | – |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
