# Per-customer deployments — design

One pipeline, two layers, one gate. A **customer** is anyone whose decisions the model should learn: a
support desk, a moderation team, a game studio. Each gets their own versioned **release**, evaluated on their own held-out
rows. A release goes live only if it beats the base model on that holdout.

```
$DJP_HOME/<id>/customer.json + data.jsonl  questions, rules in words, labelled rows
        │  split (by hash of the row; ~30 % held out, stable across machines)
        ▼
 baseline eval on holdout  ─────────────────────────────  the bar every candidate must clear
        ▼
 candidates                                               layer 1: profile, profile+calibration, base+calibration
        │                                                 layer 2: LoRA weights (+ calibration)
        ▼
 gate, per question: accuracy >= base and log-loss < base; else that question stays on base;
       refuse the release if every question stays on base
        ▼
 releases/<version>/release.json   (+ CURRENT pointer, written last, atomically)
        ▼
 serve: layer 1 → /c/<id>/v1/systemone on the shared GPU   ·   layer 2 → its own Cloud Run service
        ▼
 rollback: point CURRENT at the previous version (layer 2: Cloud Run traffic to the previous revision)
```

## Layer 1 — built and running (pipeline/djp.py)

Onboarding: `djp.py init <customer>`, `import <file> --state <col> --label <cols>` (their CSV or JSONL, their
column names), `check` (validation, no GPU), `release`. Customer data lives in `$DJP_HOME` (default
`~/.djev/customers`), never in the repo. Served at `/c/<customer>/v1/systemone`.

No training. The release is the person's rules and a few worked examples placed in front of the state, plus a
per-option calibration (a log-prior shift fitted on their training rows). The plan picks a variant per question.
Serving costs at most two parallel model calls. `python3 pipeline/djp.py release <id>` does the whole
chain in about 5 s for 40 rows.

Measured on the sample customer `acme-support` (40 rows written for this repo; 10 held out), 2026-09-25:
team accuracy 0.6 → 0.9 and urgency held at 0.7, with mean log-loss 1.81 → 0.41. On 8 tickets outside the data,
6/8 against the base model's 4/8. It fixes every login and 2FA ticket the base model sends to "technical".

Limits found: rules as prose hurt the base read (team accuracy 0.7 → 0.3 before calibration), so calibration does
most of the work. Answers on borderline tickets change between calls, because the server randomises the answer
slots on each call. A 10-row holdout moves by 0.1 per flipped row: use at least 50 held-out rows per question
before trusting a release.

## Layer 2 — trained weights (designed, not built)

When layer 1 plateaus on a person's holdout, train. Facts behind the design (checked 2026-09-25):

- The base is Google's `google/diffusiongemma-26B-A4B-it` (BF16, 51.7 GB, Apache-2.0, not gated). The model we serve
  is NVIDIA's NVFP4 quantization of it.
- vLLM's `DiffusionGemmaForConditionalGeneration` does **not** declare `SupportsLoRA` (checked on vLLM main).
  So there is no per-request adapter switching and no NVFP4 base + LoRA. **Each customer needs their own merged,
  re-quantized checkpoint and their own Cloud Run service.**
- Published trainers: NVIDIA NeMo AutoModel `diffusion_gemma_lora.yaml` and `diffusion_gemma_sft.yaml`, and Unsloth.
  A third-party tutorial reports BF16 LoRA on one H100 80 GB (reported, not checked here).

Stages, each writing into the same release record:

1. **Train**: LoRA (rank 16, attention + dense MLP, router frozen) on BF16, on one 80–96 GB GPU. The objective is
   cross-entropy on the answer-slot tokens with the template pinned and the slots randomly corrupted. That is the
   same read `/v1/systemone` does. Mix in the standard diffusion loss so general behaviour holds.
2. **Package**: merge LoRA into BF16. Quantize to NVFP4 with NVIDIA ModelOpt (`hf_ptq.py --qformat nvfp4`). Upload
   to `gs://<bucket>/customers/<id>/<version>/` with a content hash in the release record.
3. **Deploy**: `djev-cloudrun.sh` with `SERVICE=djev-<id>` and the image's `MODEL` env pointed at that prefix.
4. **Gate**: re-run the person's holdout against the deployed NVFP4 service. It must beat their live layer-1 release,
   not only the base model.

Costs and limits per trained customer:
- **GPU quota:** one GPU of quota per region per project allows one GPU service. Every trained customer needs quota
  (a request per region) or their own project. The billing account's 5-project cap also counts.
- **Idle cost:** about 35 US cents a month of storage per customer.
- **Running cost:** $3.19/h while their service is up. A cold start is about 2 minutes (122.6 s measured).
- **Training:** about 1–2 GPU-hours per customer (estimate, $3–10 at 2026-09 list prices).

The alternative to per-customer services is adding `SupportsLoRA` to vLLM's DiffusionGemma, which is unscoped.

## Layer 2 trial — prepared, not run (2026-09-25)

**Goal.** Train one customer (acme-support first, then a real one) and learn three things:
- whether trained weights beat layer 1 on the same holdout;
- what a customer model costs to produce;
- whether the cheap packaging path below works.

**Where.** A spot `g4-standard-48` VM (one RTX PRO 6000, 48 vCPU, 180 GB) in us-central1, with a 300 GB disk. This
is the same GPU as serving and has no time cap. Cloud Run jobs cap GPU tasks at 1 hour (reported). Quota filed
and granted today in djev-ouz56i: `GPUS-ALL-REGIONS` 1, spot RTX PRO 6000 us-central1 1. Spot price about
$1.77/h, on-demand about $4.50/h (third-party price list, reported).

**How.**
1. **Trainer:** Unsloth's DiffusionGemma LoRA notebook, run as a script. Its corruption step is patched so only the
   answer-slot tokens are corrupted and scored, with the template pinned. That matches the one-step read
   `/v1/systemone` does. LoRA targets are attention + dense MLP only; experts stay frozen.
2. **Packaging (the cheap path):** NVIDIA's NVFP4 checkpoint compresses only the experts. Checked today:
   `hf_quant_config.json` excludes `*self_attn*` and `*mlp*`, and the index stores `experts.N.*` with scales but
   `self_attn.*` and `mlp.*` as plain tensors. So the trained attention and MLP tensors can be copied into a copy of
   NVIDIA's checkpoint with no re-quantization. If that fails: NVIDIA ModelOpt ≥ 0.45 has a DiffusionGemma
   `nvfp4_experts_only` recipe (reported).
3. **Evaluate:** upload to `gs://<bucket>/customers/<id>/<version>/`. Point a no-traffic revision of `djev` at it
   (`MODEL=/mnt/gcs/customers/<id>/<version>`). Run the customer's holdout. Delete the revision after.

**Pre-checks, in order.** Each is cheap and stops the trial early if it fails.

| # | Risk | Pre-check |
|---|---|---|
| 1 | vLLM rejects a spliced checkpoint | Splice the *unchanged* base tensors; serve; top probabilities match the stock model on 20 rows |
| 2 | Training forward ≠ the served read | Base-model slot accuracy in the trainer equals `/v1/systemone` on the same 50 rows |
| 3 | Merge corrupts LoRA layers (Unsloth warns) | Slot logits with the adapter unmerged vs merged, in BF16 |
| 4 | LoRA reaches the experts | Print the adapter's target modules; no `experts.` names |
| 5 | Unsloth (`transformers==5.11.0`) and vLLM clash | Separate venvs; smoke-import both |
| 6 | Spot preemption | Checkpoints to GCS; resume once |

**Estimate** for one customer with about 1k rows, excluding debugging (hours: reported step speeds + inference):

| Stage | Hours low / expected / high | USD |
|---|---|---|
| VM setup, download 51.7 GB BF16 + 18.9 GB NVFP4 | 0.25 / 0.5 / 1 | |
| Pre-checks 1–5 | 0.25 / 0.5 / 1 | |
| Train (1–3 epochs) | 0.15 / 0.4 / 1 | |
| Merge + splice (or ModelOpt) | 0.1 / 0.3 / 1.5 | |
| Upload + evaluate on Cloud Run (GPU at $3.19/h) | 0.3 / 0.5 / 1 | |
| **VM total** at $1.77/h spot | **1.1 / 2.2 / 5.5 h** | **$2 / $4 / $10** |
| **Cloud Run eval** | 0.3 / 0.5 / 1 h | **$1 / $1.6 / $3.2** |
| **Trial total** | | **$3 / $6 / $13** |

Plus storage of about 90 GB for the days it is kept (under $2 a month, deleted after) and debugging time at
$1.77/h. Every repeat customer after the first costs about 1 hour of VM ($2) plus their eval.

**Budget.** The expected trial fits a small monthly GPU budget; the high case plus debugging may not.

## Open

- Minimum rows per customer for layer 2 to beat layer 1: unknown until the first spike.
- NVFP4 re-quantization of a fine-tuned DiffusionGemma with ModelOpt: untested.
- Averaging several reads per call (the noise above) versus one read: untested.
