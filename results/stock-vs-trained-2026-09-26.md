# Stock vs trained at every stage, on real labelled data (2026-09-26)

**Setup.** Five public, human-labelled sets were fetched with `pipeline/bench.py`. Each has a training pool and 1,000
fixed test rows, the same rows for every cell:
- MASSIVE English and Hindi: real voice-assistant utterances, 60 intents, CC-BY-4.0.
- CLINC150: 150 intents plus out-of-scope, CC-BY-3.0.
- banking77: 77 intents, CC-BY-4.0.
- civil_comments: toxic or not, balanced 50/50, CC0.

**Stock** means no labels: GLiNER2.5-Decide; its multilingual version, GLiNER2.5-multi; djev with 1 read; djev
averaged over 3 reads in shuffled option orders.
**Using labels**, stage by stage: calibration of a stock model, fitted on that stage's rows (up to 20 per option; 10
for CLINC); Ettin-17M fine-tuned ("tiny"); mmBERT-small (140M, multilingual) fine-tuned. Stages are nested: 5, 10,
20, 50 and all labelled rows per option. Harness: `pipeline/curve.py`. Raw cells: `data/bench/<set>/curve.jsonl`.
One standard error on 1,000 rows is about 1.5 pts at 70 %.

**Read this first.**
- **A trained model beats the best stock model by more than 2 standard errors from 50 labels per option.** That holds on 4
  of 5 sets (banking77, MASSIVE English, MASSIVE Hindi, CLINC). At 20 per option, tiny is still at or below the best
  stock model on every set. mmBERT is level with it on CLINC (71.8 % vs 69.6 %, within noise).
- **With all labels, trained wins by 18–19 pts:**
  - banking77: tiny 89.9 % vs GLiNER 70.8 %.
  - MASSIVE English: mmBERT 87.7 % vs djev@3 69.6 %.
  - MASSIVE Hindi: mmBERT 84.0 % vs djev@3 65.0 %.
  - CLINC: mmBERT 87.9 % vs djev@3 69.6 %.
  - Training takes 1–10 minutes on this Mac.
- **civil comments is the exception.** Toxicity is a 2-option judgement, and stock djev + calibration reaches 72.5 %
  with 20 labels per option. Tiny needs the whole 4,000-row pool to pass it: 81.8 %.
- **mmBERT beats Ettin-17M at every stage on the three short-text sets, by 3–12 pts.** On Hindi the gap is 7–12 pts
  (84.0 % vs 76.2 % with all labels). On civil it is mixed and within noise. It costs 4–8 times the training time: 373 s vs 79 s on 11.5k rows. This answers the open question: add
  mmBERT as a trained arm, at least for non-English customers.
- **GLiNER cannot read Hindi with English labels:** 6.1 % (English model) and 10.3 % (multilingual model). The same
  multilingual model scores 49.3 % on English. djev@3 reads Hindi at 65.0 %.
- **Averaging helps djev most when there are many options:** +17 to +26 pts over 1 read on the 60–151-option sets. On
  the 2-option set it does not help (69.6 % → 68.2 %).
- **Calibration is flat across stages.** A per-option prior shift is learned from 5 labels per option as well as from
  20. It lifts GLiNER on MASSIVE English by 17 pts (49.3 → 66.7 %), and elsewhere by 1–3 pts. It lifts 1-read djev by
  2–9 pts. That stays below djev@3 on the many-option sets, but beats it on civil (72.5 % vs 68.2 %).
- **Not run:** djev@3 + calibration. mmBERT on civil past 20 per option: at 256 tokens it grew past 11 GB on MPS, and
  I stopped it.

**Cost.** The djev cells ran 00:32–00:53 IST: about 24k calls, stock, +cal and @3. All other cells were local and
free.

## MASSIVE, English (60 intents)

| Arm | stock | 5/opt (299 rows) | 10/opt (594 rows) | 20/opt (1176 rows) | 50/opt (2831 rows) | all/opt (11514 rows) |
|---|---|---|---|---|---|---|
| gliner | 49.3 / 1.88 | – | – | – | – | – |
| gliner-multi | 49.3 / 2.42 | – | – | – | – | – |
| djev | 45.1 / 3.76 | – | – | – | – | – |
| djev@3 | 69.6 / 1.77 | – | – | – | – | – |
| gliner+cal | – | 66.0 / 1.27 | 66.4 / 1.26 | 66.7 / 1.26 | – | – |
| djev+cal | – | 53.1 / 2.65 | 53.0 / 2.66 | 53.0 / 2.66 | – | – |
| tiny | – | 23.8 / 4.40 [10 s] | 39.9 / 3.08 [9 s] | 56.2 / 2.13 [9 s] | 70.0 / 1.35 [13 s] | 83.9 / 0.75 [51 s] |
| mmbert | – | 31.8 / 4.01 [56 s] | 52.0 / 2.51 [58 s] | 64.6 / 1.74 [64 s] | 78.0 / 1.05 [88 s] | 87.7 / 0.67 [381 s] |

Best stock: 69.6 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## MASSIVE, Hindi (the same 60 intents)

| Arm | stock | 5/opt (299 rows) | 10/opt (594 rows) | 20/opt (1176 rows) | 50/opt (2831 rows) | all/opt (11514 rows) |
|---|---|---|---|---|---|---|
| gliner | 6.1 / 6.97 | – | – | – | – | – |
| gliner-multi | 10.3 / 4.65 | – | – | – | – | – |
| djev | 38.9 / 4.30 | – | – | – | – | – |
| djev@3 | 65.0 / 2.05 | – | – | – | – | – |
| gliner-multi+cal | – | 11.9 / 3.94 | 11.7 / 3.94 | 11.8 / 3.93 | – | – |
| djev+cal | – | 47.7 / 3.02 | 48.0 / 3.02 | 48.1 / 3.01 | – | – |
| tiny | – | 23.4 / 4.96 [13 s] | 35.1 / 3.78 [14 s] | 44.6 / 2.92 [15 s] | 57.9 / 1.92 [19 s] | 76.2 / 1.14 [79 s] |
| mmbert | – | 30.9 / 4.27 [56 s] | 47.4 / 3.07 [61 s] | 56.2 / 2.41 [70 s] | 69.7 / 1.51 [86 s] | 84.0 / 0.88 [373 s] |

Best stock: 65.0 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## CLINC150 (150 intents + out-of-scope)

| Arm | stock | 5/opt (755 rows) | 10/opt (1510 rows) | 20/opt (3020 rows) | 50/opt (7550 rows) | all/opt (15250 rows) |
|---|---|---|---|---|---|---|
| gliner | 63.9 / 1.83 | – | – | – | – | – |
| djev | 52.1 / 3.79 | – | – | – | – | – |
| djev@3 | 69.6 / 2.25 | – | – | – | – | – |
| gliner+cal | – | 65.1 / 1.69 | 65.1 / 1.69 | – | – | – |
| djev+cal | – | 54.0 / 3.37 | 54.1 / 3.37 | – | – | – |
| tiny | – | 35.1 / 3.80 [10 s] | 53.4 / 2.40 [10 s] | 67.2 / 1.64 [14 s] | 78.6 / 1.11 [35 s] | 84.7 / 0.65 [79 s] |
| mmbert | – | 39.9 / 3.75 [63 s] | 59.3 / 2.45 [69 s] | 71.8 / 1.45 [98 s] | 81.5 / 1.05 [244 s] | 87.9 / 0.65 [604 s] |

Best stock: 69.6 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## banking77 (77 intents)

| Arm | stock | 5/opt (385 rows) | 10/opt (770 rows) | 20/opt (1540 rows) | 50/opt (3826 rows) | all/opt (10003 rows) |
|---|---|---|---|---|---|---|
| gliner | 70.8 / 1.28 | – | – | – | – | – |
| djev | 45.3 / 4.37 | – | – | – | – | – |
| djev@3 | 63.6 / 2.28 | – | – | – | – | – |
| gliner+cal | – | 71.9 / 1.19 | 72.5 / 1.19 | 72.6 / 1.19 | – | – |
| djev+cal | – | 49.9 / 3.51 | 49.9 / 3.52 | 49.7 / 3.51 | – | – |
| tiny | – | 30.1 / 3.60 [29 s] | 51.0 / 2.27 [14 s] | 67.3 / 1.28 [15 s] | 83.3 / 0.65 [26 s] | 89.9 / 0.39 [69 s] |

Best stock: 70.8 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## civil_comments (toxic or not, balanced)

| Arm | stock | 5/opt (10 rows) | 10/opt (20 rows) | 20/opt (40 rows) | 50/opt (100 rows) | all/opt (4000 rows) |
|---|---|---|---|---|---|---|
| gliner | 64.8 / 1.06 | – | – | – | – | – |
| djev | 69.6 / 1.20 | – | – | – | – | – |
| djev@3 | 68.2 / 1.24 | – | – | – | – | – |
| gliner+cal | – | 67.9 / 0.75 | 68.0 / 0.75 | 68.1 / 0.75 | – | – |
| djev+cal | – | 71.3 / 0.71 | 71.7 / 0.69 | 72.5 / 0.68 | – | – |
| tiny | – | 52.0 / 2.52 [2 s] | 55.0 / 2.02 [3 s] | 59.6 / 1.34 [6 s] | 68.7 / 1.15 [17 s] | 81.8 / 0.85 [94 s] |
| mmbert | – | 55.8 / 1.32 [7 s] | 52.4 / 1.29 [12 s] | 61.5 / 0.94 [26 s] | – | – |

Best stock: 69.6 %. A trained model first beats it by more than 2 standard errors at: all labels per option.

Cells: accuracy % / log-loss; training seconds in brackets. "–" means not run: stock arms have no stages,
calibration stops at 20 (CLINC 10) per option, and mmBERT on civil stops at 20.
