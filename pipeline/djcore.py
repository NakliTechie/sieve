"""Per-customer release: find it, turn a state into a /v1/systemone request, calibrate the answer.

Shared by the release pipeline (djp.py) and the serving route (demo/serve.py), so what is evaluated is what is served.
Stdlib only.

A customer lives in $DJP_HOME/<id>/ (default data/customers in this repo, gitignored so customer data never reaches git);
the repo's pipeline/examples/<id>/ holds made-up samples and is searched second.
  customer.json   id, questions (systemone format), context (their rules, in words), few_shot (count)
  data.jsonl      labelled rows: {"state": "...", "answers": {"<question>": "<option>", ...}}
  releases/<version>/release.json   written by `djp.py release`; CURRENT names the live version
"""
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(os.environ.get("SIEVE_DATA", Path(__file__).resolve().parent.parent / "data"))  # gitignored
HOME = Path(os.environ.get("DJP_HOME", DATA / "customers"))
EXAMPLES = Path(__file__).with_name("examples")
HOLDOUT_TENTHS = 3  # rows whose state hashes into 0..2 of 10 are held out: stable across runs and machines


def customer_dir(cid):
    """-> the customer's folder ($DJP_HOME first, then examples), or None."""
    for root in (HOME, EXAMPLES):
        if (root / cid / "customer.json").exists():
            return root / cid
    return None


def all_customers():
    return sorted({d.name: d for root in (EXAMPLES, HOME) if root.exists()
                   for d in root.iterdir() if (d / "customer.json").exists()}.values(), key=lambda d: d.name)


def load_customer(cid):
    d = customer_dir(cid)
    customer = json.loads((d / "customer.json").read_text())
    data = d / "data.jsonl"
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()] if data.exists() else []
    return customer, rows


def _bucket(row):
    return int(hashlib.sha1(row["state"].encode()).hexdigest(), 16) % 10


def split(rows):
    """-> (few_shot pool + calibration rows, holdout rows); ordered by hash so the split never drifts."""
    rows = sorted(rows, key=lambda r: hashlib.sha1(r["state"].encode()).hexdigest())
    return [r for r in rows if _bucket(r) >= HOLDOUT_TENTHS], [r for r in rows if _bucket(r) < HOLDOUT_TENTHS]


def render_state(profile, state):
    """The customer's rules and worked examples go in front of the state; the answer template stays the questions'."""
    if not profile:
        return state
    parts = [profile["context"]] if profile.get("context") else []
    if profile.get("examples"):
        parts.append("Examples:\n" + "\n".join(
            f"- {ex['state']} -> " + "; ".join(f"{q}: {a}" for q, a in ex["answers"].items())
            for ex in profile["examples"]))
    parts.append("Now answer for this one:\n" + state)
    return "\n\n".join(parts)


def ask(base_url, questions, state, timeout=300, tries=3):
    """One /v1/systemone call; retries a 5xx (seen once, transient, under parallel load) with backoff."""
    body = json.dumps({"state": state, "questions": questions}).encode()
    for i in range(tries):
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == tries - 1:
                raise
            time.sleep(2 ** i)


def calibrate(probs, bias):
    """probs {option: p}, bias {option: b} -> softmax(log p + b). An empty bias is the identity."""
    if not bias:
        return dict(probs)
    logits = {k: math.log(max(p, 1e-6)) + bias.get(k, 0.0) for k, p in probs.items()}
    m = max(logits.values())
    z = sum(math.exp(v - m) for v in logits.values())
    return {k: math.exp(v - m) / z for k, v in logits.items()}


def answer(base_url, release, state):
    """Serve one call for a released customer: each question on its planned prompt (at most two calls, in parallel),
    then its calibration."""
    groups = {}
    for q, spec in release["plan"].items():
        groups.setdefault(spec["use_profile"], {})[q] = release["questions"][q]
    with ThreadPoolExecutor(len(groups)) as ex:
        raws = list(ex.map(lambda g: ask(base_url, g[1], render_state(release["profile"] if g[0] else None, state)),
                           groups.items()))
    out, timing = {}, []
    for raw in raws:
        timing.append(raw.get("diagnostics", {}).get("timing", {}).get("total_ms"))
        for q, a in raw.get("answers", {}).items():
            p = calibrate(a.get("probabilities", {}), release["plan"][q]["calibration"])
            best = max(p, key=p.get)
            out[q] = {"choice": best, "probabilities": {k: round(v, 6) for k, v in p.items()}, "confidence": round(p[best], 6)}
    return {"model": raws[0].get("model"), "customer": release["customer"], "release": release["version"],
            "answers": out, "diagnostics": {"timing": {"total_ms": max(t for t in timing if t is not None) if any(timing) else None},
                                            "calls": len(raws)}}


def current_release(cid):
    d = customer_dir(cid)
    if d is None:
        return None
    cur = d / "releases" / "CURRENT"
    if not cur.exists():
        return None
    return json.loads((d / "releases" / cur.read_text().strip() / "release.json").read_text())


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
