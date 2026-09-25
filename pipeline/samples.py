#!/usr/bin/env python3
"""Build stand-in customers from public Hugging Face datasets, for testing the pipeline without real customer data.

  python3 pipeline/samples.py list                 # the recipes, with source, license and whether the data is real
  python3 pipeline/samples.py build <recipe> [N]   # fetch ~N rows (default 250) -> $DJP_HOME/<recipe>/source.csv,
                                                   # customer.json; then run: djp.py import <recipe> ... (printed)

Rows come from the Hugging Face datasets-server API (no download of the full dataset, no account). Sampling is
deterministic (evenly spaced windows) and balanced over one label per recipe. Stdlib only. The data lands in
$DJP_HOME, never in the repo: some sources are non-commercial.
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djcore as P  # noqa: E402

API = "https://datasets-server.huggingface.co"
PRIORITY = {"very_low": "low", "low": "low", "medium": "medium", "high": "high", "critical": "high"}
QUEUES = ["Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments",
          "Returns and Exchanges"]

RECIPES = {
    "helpdesk": {
        "dataset": "Tobi-Bueck/customer-support-tickets", "license": "cc-by-nc-4.0", "real": False,
        "note": "synthetic IT/customer-support tickets; labels are noisy",
        "balance": "queue",
        "row": lambda x: None if x["queue"] not in QUEUES or x["language"] != "en" else {
            "text": ((x.get("subject") or "") + "\n" + (x.get("body") or "")).strip()[:1200],
            "queue": x["queue"], "priority": PRIORITY.get(x["priority"]), "type": x["type"]},
        "labels": ["queue", "priority", "type"],
        "questions": {
            "queue": "Which support queue should handle this ticket?",
            "priority": "How urgent is this ticket?",
            "type": "What kind of ticket is this: an incident, a request, a problem or a change?"},
        "context": "Route tickets to the queue that owns the product or process. Billing and Payments handles invoices "
                   "and charges; Returns and Exchanges handles returns; IT Support handles internal IT and accounts; "
                   "Technical Support handles faults in the product; Product Support handles how-to questions; "
                   "Customer Service handles everything else. Priority is high when work is blocked or data or money "
                   "is at risk, low for information requests.",
    },
    "shop": {
        "dataset": "bitext/Bitext-customer-support-llm-chatbot-training-dataset", "license": "cdla-sharing-1.0",
        "real": False, "note": "synthetic e-commerce customer messages", "balance": "category",
        "row": lambda x: {"text": x["instruction"], "category": x["category"].lower()},
        "labels": ["category"],
        "questions": {"category": "What is this customer message about?"},
        "context": "Messages about logging in, signing up, passwords or editing a profile are account. Getting "
                   "money back is refund; paying or payment methods is payment; bills and receipts are invoice. "
                   "Where a parcel is or when it arrives is delivery; shipping addresses and options are shipping.",
    },
    "moderation": {
        "dataset": "google/civil_comments", "license": "cc0-1.0", "real": True,
        "note": "real online comments; toxicity and insult scored by human raters (fraction of raters)",
        # balanced on toxic: the natural rate (~8 %) would leave few positives; unclear rows (0.1-0.5) are skipped
        "balance": "toxic",
        "row": lambda x: None if 0.1 <= x["toxicity"] < 0.5 else {"text": x["text"][:1200], "toxic": "yes" if x["toxicity"] >= 0.5 else "no",
                          "insult": "yes" if x["insult"] >= 0.5 else "no"},
        "labels": ["toxic", "insult"],
        "questions": {"toxic": "Is this comment toxic: rude, disrespectful or unreasonable enough to make someone leave?",
                      "insult": "Does this comment insult a person or group?"},
        "context": "We moderate a news site's comments. Strong disagreement, criticism of public figures' actions "
                   "and swearing that is not aimed at anyone are allowed. Name-calling, slurs, and attacks on a "
                   "person or group are toxic.",
    },
}


def fetch(dataset, offset, length=100, tries=5):
    q = {"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": length}
    url = f"{API}/rows?" + urllib.parse.urlencode(q)
    for i in range(tries):
        time.sleep(1)  # the public API rate-limits (HTTP 429) bursts
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.loads(r.read())
            return [x["row"] for x in d.get("rows", [])], d.get("num_rows_total", 0)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(20 * (i + 1))
        except (TimeoutError, OSError):
            if i == tries - 1:
                raise


def sample(recipe, n, reads=60):
    """Read `reads` evenly spaced 100-row windows across the whole split (datasets are often stored sorted by
    label), keep rows the recipe accepts, then take round-robin over the balance label."""
    ds, bal = recipe["dataset"], recipe["balance"]
    _, total = fetch(ds, 0, 1)
    groups = {}
    for k in range(reads):
        rows, _ = fetch(ds, int((total - 100) * k / max(reads - 1, 1)))
        for y in (recipe["row"](x) for x in rows):
            if y and all(y.get(c) for c in recipe["labels"]):
                groups.setdefault(y[bal], []).append(y)
        if len(groups) >= 2 and all(len(g) >= n for g in groups.values()):
            break
    take, i = [], 0
    while len(take) < n and any(groups.values()):
        g = sorted(groups)[i % len(groups)]
        if groups[g]:
            take.append(groups[g].pop(0))
        i += 1
    return take


def build(name, n):
    rec = RECIPES[name]
    rows = sample(rec, n)
    d = P.HOME / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "source.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text"] + rec["labels"])
        w.writeheader()
        w.writerows(rows)
    crit = {c: sorted({r[c] for r in rows}) for c in rec["labels"]}
    P.write_atomic(d / "customer.json", json.dumps({
        "id": name, "description": f"Stand-in customer from {rec['dataset']} ({rec['license']}; "
                                   f"{'real' if rec['real'] else 'synthetic'}: {rec['note']}).",
        "questions": {c: {"type": "choice", "instructions": rec["questions"][c], "criteria": crit[c]} for c in rec["labels"]},
        "context": rec["context"], "few_shot": 6}, indent=2) + "\n")
    counts = {c: {v: sum(r[c] == v for r in rows) for v in crit[c]} for c in rec["labels"]}
    print(f"verdict=BUILT customer={name} rows={len(rows)} source={rec['dataset']} license={rec['license']} "
          f"real={rec['real']} counts={json.dumps(counts)}")
    print(f"next: python3 pipeline/djp.py import {name} {d}/source.csv --state text --label {','.join(rec['labels'])}")


def main(argv):
    if argv[:1] == ["list"]:
        for k, r in RECIPES.items():
            print(f"{k:11} {r['dataset']:62} {r['license']:17} {'real' if r['real'] else 'synthetic':9} {r['note']}")
        return 0
    if argv[:1] == ["build"] and len(argv) > 1 and argv[1] in RECIPES:
        build(argv[1], int(argv[2]) if len(argv) > 2 else 250)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
