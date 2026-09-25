<h1 align="center">sieve</h1>

**A decision classifier per customer: each question starts on a model that needs no training, moves to a 17M model
trained on the customer's own labels in seconds, and switches only when it wins on their held-out rows.**

Python 3.12 on a Mac (Apple MPS) or any CPU. Customer data stays on your disk, in a gitignored folder. The one
optional cloud part is the djev GPU service.

## Install

| Platform | Command |
|---|---|
| macOS / Linux, [uv](https://docs.astral.sh/uv/) | `git clone https://github.com/NakliTechie/sieve && cd sieve && uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install "gliner2[train]"` |

Build a stand-in customer from public data, release it on the two local arms, and serve it:

```bash
python3 pipeline/samples.py build shop      # prints the import command; run it
.venv/bin/python pipeline/djp.py release shop --arms gliner,tiny
.venv/bin/python pipeline/serve.py &        # port 8090
curl -s localhost:8090/c/shop/v1/systemone -d '{"state": "where is my parcel? it was due monday"}'
```

Every answer names the backend that produced it. The first release downloads GLiNER2.5-Decide (1.8 GB) and
Ettin-17M (130 MB) from Hugging Face. No account is needed.

## Why

Your team routes tickets or flags comments by hand. A zero-shot model gets you started. But for your categories it
tops out around 60–70 %, and it answers differently when you reorder the options. A small model trained on your
own labels would do better. You still need a way to know when it does, per question, without trusting a single
average.

sieve runs three model families on every question and scores each on rows it never trained on:
- **djev**: DiffusionGemma-Jev over `/v1/systemone` on a Cloud Run GPU, with or without your rules and calibration.
- **GLiNER2.5-Decide**: 340M parameters, local.
- **Ettin-17M**: fine-tuned on your labels once the rarest option has 20 of them.

A candidate replaces plain djev only if its holdout accuracy is at least djev's and its log-loss is lower. The
winner is served exactly as it was scored.

**Use something else if** your labels number in the thousands and you want one model:
[SetFit](https://github.com/huggingface/setfit) or a plain fine-tune in
[transformers](https://huggingface.co/docs/transformers/tasks/sequence_classification) is simpler.
[GLiNER2](https://github.com/fastino-ai/GLiNER2) alone is enough if you have no labels at all.
[TypeLLM](https://github.com/TypeLLM/TypeLLM) fits if you want an LLM's answer with permutation averaging.

## What it found (2026-09-25)

- On the public fast-decisions benchmark, GLiNER beats djev: 67.3 % vs 62.6 %.
- Reversing the option order flips 25 % of djev's answers. Asking the same thing twice flips 11.6 %.
- On stand-in customers, the gate chose tiny for shop: 70.8 % → 90.3 %, 72 held-out rows, p = 0.007.
- For helpdesk type, it chose djev with rules and calibration: 64.1 % → 75.6 %, p = 0.02.
- Four other questions moved within noise.
- Details: [results/](results/).

## Adding labels

```bash
.venv/bin/python pipeline/djp.py label shop new-rows.jsonl   # {"state": "...", "answers": {"category": "refund"}}
```

This appends the rows, retrains tiny, re-runs the gate, and moves the live release only on a pass. On shop it took
5.2 s. Running it again with the same file changes nothing. `rollback` returns to the previous passing release.

## Commands

```bash
python3 pipeline/djp.py status [--json]                        # every customer: live release, accuracy vs bar, backend per question
python3 pipeline/djp.py init <c>                               # new customer folder in data/customers
python3 pipeline/djp.py import <c> <file> --state <col> --label <cols>   # their CSV/JSONL -> data.jsonl (replaces)
python3 pipeline/djp.py check <c>                              # validate; no model calls
.venv/bin/python pipeline/djp.py eval <c> [--arms djev,gliner,tiny]      # every candidate on the holdout; no release
.venv/bin/python pipeline/djp.py release <c> [--arms ...] [--force]      # gate -> release -> CURRENT
.venv/bin/python pipeline/djp.py label <c> <file>                        # append labels -> retrain -> gate -> release
python3 pipeline/djp.py rollback <c> [version]                 # CURRENT -> an earlier passing release
.venv/bin/python pipeline/serve.py                             # POST /c/<c>/v1/systemone, GET /c/<c>
.venv/bin/python pipeline/compare.py [reread]                  # table of live releases with sign tests | bar re-read noise
python3 pipeline/samples.py build shop|helpdesk|moderation     # stand-in customers from public HF data
.venv/bin/python pipeline/ab.py run [--reads K] && python3 pipeline/ab.py report   # djev vs GLiNER, option order, averaging
.venv/bin/python pipeline/tiny.py banking77 | customer <c>     # tiny benchmark
```

Agents: every command is declared in [tools.json](tools.json) with its input schema, cost class and
`delegable: agent | person-only`. Labels are person-only. [SPEC.md §0](SPEC.md) is the agent contract.

The djev arm needs `gcloud run services proxy djev --region us-central1 --project djev-ouz56i --port 8081`. The GPU
costs $3.19/h while it is up. `--arms gliner,tiny` keeps everything local.

## Verify it yourself

```bash
python3 -m unittest discover tests                             # gate, idempotent label, crash mid-release, refusal, manifest parity
.venv/bin/python pipeline/compare.py                           # rebuilds each release's predictions; asserts they equal the gated numbers
```

The gate refuses a release when no question beats the bar. `CURRENT` then stays where it was, with exit 7.
A release killed midway leaves the previous one live. Serving shop's tiny model on CPU reproduced its gated
holdout accuracy, 90.28 % on 72 rows.

## License

Private research repo; no license granted yet. · [SPEC.md](SPEC.md) · [TRAINING.md](TRAINING.md) (layer-2 design) ·
[results/](results/)
