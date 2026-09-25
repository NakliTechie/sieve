# sieve

> A decision classifier per customer: answer from a model that needs no training on day one, then train a 17M
> model on the customer's own labels in seconds, and switch only when it wins on their held-out rows.

Research carried over from [djev-demo](https://github.com/NakliTechie/djev-demo) (commit `51c3dc8`, 2026-09-25).
A customer brings questions with fixed options (route this ticket, flag this comment) and labelled examples.
Three model families compete per question, on that customer's held-out rows:

- **djev**: DiffusionGemma-Jev over `/v1/systemone` (Cloud Run service `djev`, GCP project `djev-ouz56i`), with no training.
- **GLiNER2.5-Decide** (340M, Apache-2.0), with no training, on a laptop.
- **Ettin-17M** (MIT), fine-tuned per customer in 2–60 seconds on a laptop GPU.

Measured on 2026-09-25 ([results/](results/)): on the public fast-decisions benchmark GLiNER beats djev (67.3 % vs
62.6 %). Changing only the option order flips 25 % of djev's answers (11.6 % flip with nothing changed), against 3 %
for GLiNER. Ettin-17M reaches 90.0 % on banking77 after 42.6 s of training, and beats both zero-shot models once a
customer has about 20 labels per option.

## Install

```bash
git clone https://github.com/NakliTechie/sieve && cd sieve
uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install "gliner2[train]"
```

Data lives in `data/` (gitignored): customers in `data/customers` (`$DJP_HOME`), benchmarks, A/B rows, the training log. Customer data never goes in git. The djev arm also needs
`gcloud run services proxy djev --region us-central1 --project djev-ouz56i --port 8081`.

## Commands

```bash
python3 pipeline/djp.py status | init <c> | import <c> <file> --state <col> --label <cols> | check <c> | release <c>
python3 pipeline/samples.py build shop|helpdesk|moderation      # stand-in customers from public HF datasets
.venv/bin/python pipeline/ab.py run && python3 pipeline/ab.py report   # djev vs GLiNER + option-order test
.venv/bin/python pipeline/tiny.py banking77 | customer <c>             # train-on-the-fly Ettin-17M
```

## Next steps

- One release per customer that can pick its model per question: djev, GLiNER, or tiny. It starts on the
  zero-shot models, then retrains tiny as labels arrive, and promotes only through the per-question holdout gate.
- Test averaging over shuffled option orders for djev, the published fix for the order effect (TypeLLM).
- A first real customer's data (~170+ labelled rows).
