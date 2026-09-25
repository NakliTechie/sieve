#!/usr/bin/env python3
"""Markdown table of the live releases: per question, the backend the gate chose, holdout accuracy against the bar,
every candidate's accuracy, and whether the gain is outside noise.

  .venv/bin/python pipeline/compare.py [customer ...]      # default: every customer with a live release

Noise: a paired sign test on the holdout rows where the chosen model and the bar disagree about correctness
(exact binomial, two-sided). Per-row predictions are rebuilt from the release: djev and gliner from the read cache
(the same reads the gate saw; no model calls), tiny from the weights saved in the release. If data/
releases-reread-*.json exists (a second, uncached djev read of the same holdouts), the bar's re-read accuracy is shown.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import djcore as P  # noqa: E402

URL = os.environ.get("DJEV_URL", "http://localhost:8081")


def sign_p(b, c):
    """Two-sided exact binomial p for b successes vs c failures under p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def rows_for(release, hold):
    """-> {question: [chosen prediction]}, {question: [bar prediction]} over the holdout, from caches and weights."""
    cid, qs, plan = release["customer"], release["questions"], release["plan"]
    states = [r["state"] for r in hold]
    bar = release.get("bar", "base")
    reads = {}

    def read(backend, use_profile):
        k = (backend, use_profile)
        if k not in reads:
            if backend == "djev":
                reads[k] = B.djev_read(URL, qs, release["profile"] if use_profile else None, states, B.cache_for(cid, "djev"))
            else:
                reads[k] = B.gliner_read(qs, states, B.cache_for(cid, "gliner"))
        return reads[k]
    chosen, barp = {}, {}
    rdir = P.customer_dir(cid) / "releases" / release["version"]
    for q, p in plan.items():
        if p["backend"] == "tiny":
            m, tok = B._tiny_model(str(rdir / p["path"]))
            raw = [{q: x} for x in B.tiny_proba(m, tok, states)]
        else:
            raw = read(p["backend"], p["use_profile"])
        pick = lambda pr, cal: max((c := P.calibrate(pr[q], cal)), key=c.get)
        chosen[q] = [pick(x, p["calibration"]) for x in raw]
        barp[q] = [pick(x, {}) for x in read("djev" if bar == "base" else "gliner", False)]
    return chosen, barp


def main(argv):
    cids = argv or [d.name for d in P.all_customers() if P.current_release(d.name)]
    reread = {}
    for f in sorted(P.DATA.glob("releases-reread-*.json")):
        reread = json.loads(f.read_text())
    cand_names = ["base+cal", "profile", "profile+cal", "gliner", "gliner+cal", "tiny"]
    print("| Customer · question | n | Options | Chosen | Chosen acc | Bar acc (plain djev) | Bar re-read | Δ pts | "
          "Chosen right / bar right (discordant rows) | Sign test p | " + " | ".join(cand_names) + " |")
    print("|---" * (10 + len(cand_names)) + "|")
    for cid in cids:
        r = P.current_release(cid)
        _, rows = P.load_customer(cid)
        _, hold = P.split(rows)
        chosen, barp = rows_for(r, hold)
        for q, p in r["plan"].items():
            truth = [x["answers"][q] for x in hold]
            b = sum(c == t and d != t for c, d, t in zip(chosen[q], barp[q], truth))
            c = sum(d == t and cc != t for cc, d, t in zip(chosen[q], barp[q], truth))
            acc = sum(x == t for x, t in zip(chosen[q], truth)) / len(hold)
            bar_acc = r["eval"][r.get("bar", "base")][q]["accuracy"]
            assert abs(acc - r["released_eval"][q]["accuracy"]) < 1e-3, (cid, q, acc, r["released_eval"][q]["accuracy"])
            rr = reread.get(f"{cid}:{q}", {}).get("reread")
            cands = [f"{100 * r['eval'][n][q]['accuracy']:.1f}" if q in r["eval"].get(n, {}) else "–" for n in cand_names]
            tiny = r.get("tiny", {}).get(q, {})
            if tiny and not tiny.get("eligible"):
                cands[-1] = f"– ({tiny['least_per_option']}/opt)"
            print(f"| {cid} · {q} | {len(hold)} | {len(r['questions'][q]['criteria'])} | {p['backend']} ({p['variant']}) "
                  f"| {100 * acc:.1f} | {100 * bar_acc:.1f} | {f'{100 * rr:.1f}' if rr is not None else '–'} "
                  f"| {round(100 * (acc - bar_acc), 1) + 0.0:+.1f} | {b} / {c} | {sign_p(b, c):.3g} | " + " | ".join(cands) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
