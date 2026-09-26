# Seeds and a keyword baseline: how much of the gain is the model? (2026-09-26)

**Setup.** Two additions to `pipeline/curve.py`, on the 8 datasets from `results/stock-vs-trained-2026-09-26.md` and
`results/hinglish-2026-09-26.md`. The test rows are unchanged.
1. **Seeds.** Seeds 1 and 2 join last night's seed 0. A seed changes both which labelled rows are drawn and how
   training starts. Cells show mean ± standard deviation over the seeds run.
2. **TF-IDF + logistic regression.** Word 1–2 grams plus character 2–5 grams, logistic regression (C = 10), 3 seeds
   at every stage. It runs on CPU in 0–16 s.
Also recorded: "k unused", the number of test intents a model never predicts (a collapse check). Local only,
13:46–15:17 IST. mmBERT was not re-run on civil (memory limit at 256 tokens) or at the all-labels stage.

**Read this first.**
- **Seed noise is about 1 pt.** The median standard deviation across seeds is 0.6 pts for TF-IDF, 1.0 for Ettin and
  1.2 for mmBERT. The worst cell is 4.6 pts (Ettin, Hinglish sentiment at 20 per class). Last night's single-seed
  cells hold to about ±2 pts; small-label transformer cells are the noisiest.
- **On intent routing, the keyword model is the one to beat below 50 labels per intent.** At 20 per intent,
  TF-IDF beats mmBERT on all 6 intent sets, by 3.8–20.7 pts, and Ettin by more. At 50 per intent it ties mmBERT on 3
  sets (within 1 pt), leads on both Hinglish-TOP sets (+10.3, +4.0) and trails on CLINC (−2.1).
- **With all labels, mmBERT adds 1–4 pts over TF-IDF on 4 intent sets** (CLINC, MASSIVE English and Hindi,
  Hinglish-TOP English). TF-IDF leads on Hinglish-TOP, and on banking77 against Ettin (mmBERT not run there). So the gain over zero-shot comes mostly from training on the customer's labels, not from a transformer.
- **The keyword model passes zero-shot earliest:** by more than 2 standard errors at 20 labels per intent on banking77
  and CLINC, and at 50 on MASSIVE English and Hinglish-TOP (at 20 it ties on MASSIVE English, 70.3 vs 69.6). On
  Hinglish-TOP the transformers needed all the labels.
- **Judgement tasks are the exception.** On civil comments and Hinglish sentiment the transformers beat TF-IDF, and
  zero-shot beats all of them until the full label pool (civil) or at every stage (Hinglish sentiment).
- **TF-IDF drops rare intents.** On Hinglish-TOP with all labels, it never predicts 11 of 57 intents (8 of 57 on the
  English twin). Its accuracy is still the highest there, but a customer with rare, important intents needs a check.

## TF-IDF vs the transformers at 20 and 50 labels per intent

| Dataset | Stage | TF-IDF | Ettin | mmBERT | TF-IDF − mmBERT |
|---|---|---|---|---|---|
| banking77 | 20/intent | 80.2 ± 0.5 | 67.2 ± 3.8 | 72.0 ± 1.1 | +8.2 |
| banking77 | 50/intent | 87.5 ± 0.4 | 83.2 ± 0.9 | 86.6 ± 0.7 | +0.9 |
| clinc | 20/intent | 75.8 ± 0.6 | 66.5 ± 1.0 | 72.0 ± 0.7 | +3.8 |
| clinc | 50/intent | 79.4 ± 0.3 | 77.8 ± 0.7 | 81.5 ± 0.2 | -2.1 |
| massive-en | 20/intent | 70.3 ± 1.3 | 57.1 ± 1.0 | 65.2 ± 0.6 | +5.0 |
| massive-en | 50/intent | 77.6 ± 0.4 | 70.6 ± 0.9 | 77.6 ± 1.4 | +0.0 |
| massive-hi | 20/intent | 62.0 ± 0.4 | 45.7 ± 1.2 | 56.7 ± 2.2 | +5.2 |
| massive-hi | 50/intent | 70.0 ± 1.0 | 57.0 ± 0.9 | 69.2 ± 1.6 | +0.9 |
| hinglish-top | 20/intent | 74.3 ± 0.7 | 53.8 ± 0.4 | 53.6 ± 3.7 | +20.7 |
| hinglish-top | 50/intent | 80.7 ± 0.3 | 67.8 ± 1.0 | 70.4 ± 0.5 | +10.3 |
| hinglish-top-en | 20/intent | 78.3 ± 0.8 | 62.6 ± 2.3 | 68.8 ± 2.3 | +9.5 |
| hinglish-top-en | 50/intent | 84.2 ± 0.8 | 74.7 ± 0.8 | 80.2 ± 1.0 | +4.0 |
| civil | 20/intent | 55.5 ± 2.6 | 60.7 ± 1.8 | 61.5 | -6.0 |
| civil | 50/intent | 60.6 ± 1.7 | 65.2 ± 3.4 | – | – |
| hinglish-yt | 20/intent | 45.6 ± 0.8 | 46.5 ± 4.6 | 49.4 ± 2.6 | -3.8 |
| hinglish-yt | 50/intent | 52.6 ± 0.6 | 51.4 ± 1.7 | 51.3 ± 2.3 | +1.2 |

## banking77

Test rows: 1000 (the same rows for every cell; one standard error is about 1.6 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (385 rows) | 10/opt (770 rows) | 20/opt (1540 rows) | 50/opt (3826 rows) | all/opt (10003 rows) |
|---|---|---|---|---|---|---|
| gliner | 70.8 | – | – | – | – | – |
| djev | 45.3 | – | – | – | – | – |
| djev@3 | 63.6 | – | – | – | – | – |
| gliner+cal | – | 71.9 | 72.5 | 72.6 | – | – |
| djev+cal | – | 49.9 | 49.9 | 49.7 | – | – |
| tfidf | – | 62.8 ± 2.0 [0 s] | 72.8 ± 1.8 [1 s] | 80.2 ± 0.5 [1 s] | 87.5 ± 0.4 [3 s] | 91.2 ± 0.0 [9 s] |
| tiny | – | 28.3 ± 1.6 [15 s] | 48.6 ± 3.6 [10 s] | 67.2 ± 3.8 [11 s] | 83.2 ± 0.9 [19 s] | 90.3 ± 0.4 [50 s] |
| mmbert | – | 29.5 ± 5.4 [65 s] (1 unused) | 49.5 ± 9.3 [64 s] | 72.0 ± 1.1 [75 s] | 86.6 ± 0.7 [132 s] | – |

Best stock: 70.8 %. A trained model first beats it by more than 2 standard errors at: 20 labels per option.

## clinc

Test rows: 1000 (the same rows for every cell; one standard error is about 1.6 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (755 rows) | 10/opt (1510 rows) | 20/opt (3020 rows) | 50/opt (7550 rows) | all/opt (15250 rows) |
|---|---|---|---|---|---|---|
| gliner | 63.9 | – | – | – | – | – |
| djev | 52.1 | – | – | – | – | – |
| djev@3 | 69.6 | – | – | – | – | – |
| gliner+cal | – | 65.1 | 65.1 | – | – | – |
| djev+cal | – | 54.0 | 54.1 | – | – | – |
| tfidf | – | 63.4 ± 1.0 [0 s] (1 unused) | 70.9 ± 0.9 [2 s] | 75.8 ± 0.6 [3 s] | 79.4 ± 0.3 [8 s] | 87.0 ± 0.0 [17 s] |
| tiny | – | 32.3 ± 2.4 [7 s] (2 unused) | 51.4 ± 1.7 [8 s] (1 unused) | 66.5 ± 1.0 [10 s] | 77.8 ± 0.7 [26 s] | 84.8 ± 0.3 [58 s] |
| mmbert | – | 41.6 ± 1.6 [63 s] (2 unused) | 59.0 ± 0.8 [68 s] (1 unused) | 72.0 ± 0.7 [97 s] | 81.5 ± 0.2 [241 s] | 87.9 [604 s] |

Best stock: 69.6 %. A trained model first beats it by more than 2 standard errors at: 20 labels per option.

## massive-en

Test rows: 1000 (the same rows for every cell; one standard error is about 1.6 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (299 rows) | 10/opt (594 rows) | 20/opt (1176 rows) | 50/opt (2831 rows) | all/opt (11514 rows) |
|---|---|---|---|---|---|---|
| gliner | 49.3 | – | – | – | – | – |
| gliner-multi | 49.3 | – | – | – | – | – |
| djev | 45.1 | – | – | – | – | – |
| djev@3 | 69.6 | – | – | – | – | – |
| gliner+cal | – | 66.0 | 66.4 | 66.7 | – | – |
| djev+cal | – | 53.1 | 53.0 | 53.0 | – | – |
| tfidf | – | 52.9 ± 1.4 [0 s] | 62.2 ± 0.3 [0 s] | 70.3 ± 1.3 [1 s] | 77.6 ± 0.4 [2 s] (1 unused) | 85.7 ± 0.0 [14 s] (1 unused) |
| tiny | – | 24.6 ± 1.3 [7 s] | 41.2 ± 1.1 [7 s] | 57.1 ± 1.0 [7 s] | 70.6 ± 0.9 [10 s] (1 unused) | 84.3 ± 0.5 [38 s] |
| mmbert | – | 32.4 ± 1.6 [57 s] | 50.1 ± 2.3 [60 s] | 65.2 ± 0.6 [67 s] | 77.6 ± 1.4 [90 s] | 87.7 [381 s] |

Best stock: 69.6 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## massive-hi

Test rows: 1000 (the same rows for every cell; one standard error is about 1.6 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (299 rows) | 10/opt (594 rows) | 20/opt (1176 rows) | 50/opt (2831 rows) | all/opt (11514 rows) |
|---|---|---|---|---|---|---|
| gliner | 6.1 | – | – | – | – | – |
| gliner-multi | 10.3 | – | – | – | – | – |
| djev | 38.9 | – | – | – | – | – |
| djev@3 | 65.0 | – | – | – | – | – |
| gliner-multi+cal | – | 11.9 | 11.7 | 11.8 | – | – |
| djev+cal | – | 47.7 | 48.0 | 48.1 | – | – |
| tfidf | – | 40.5 ± 1.5 [0 s] | 52.2 ± 2.1 [0 s] | 62.0 ± 0.4 [1 s] | 70.0 ± 1.0 [3 s] (1 unused) | 79.6 ± 0.0 [16 s] (1 unused) |
| tiny | – | 24.3 ± 1.4 [10 s] | 34.1 ± 1.1 [11 s] | 45.7 ± 1.2 [12 s] | 57.0 ± 0.9 [16 s] (1 unused) | 77.5 ± 1.5 [63 s] (2 unused) |
| mmbert | – | 29.5 ± 2.6 [57 s] (1 unused) | 44.0 ± 3.4 [61 s] | 56.7 ± 2.2 [69 s] | 69.2 ± 1.6 [89 s] | 84.0 [373 s] |

Best stock: 65.0 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## hinglish-top

Test rows: 1000 (the same rows for every cell; one standard error is about 1.6 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (285 rows) | 10/opt (519 rows) | 20/opt (884 rows) | 50/opt (1724 rows) | all/opt (4378 rows) |
|---|---|---|---|---|---|---|
| gliner | 49.6 | – | – | – | – | – |
| gliner-multi | 46.4 | – | – | – | – | – |
| djev | 66.0 | – | – | – | – | – |
| djev@3 | 76.6 | – | – | – | – | – |
| gliner+cal | – | 51.5 | 51.2 | 52.6 | – | – |
| tfidf | – | 56.5 ± 0.3 [0 s] (1 unused) | 65.8 ± 1.2 [0 s] (1 unused) | 74.3 ± 0.7 [1 s] (4 unused) | 80.7 ± 0.3 [2 s] (7 unused) | 84.4 ± 0.0 [5 s] (11 unused) |
| tiny | – | 28.5 ± 1.4 [7 s] (1 unused) | 42.2 ± 0.7 [8 s] | 53.8 ± 0.4 [8 s] (1 unused) | 67.8 ± 1.0 [8 s] (2 unused) | 81.0 ± 0.7 [16 s] (4 unused) |
| mmbert | – | 27.3 ± 1.9 [59 s] | 37.1 ± 2.2 [66 s] | 53.6 ± 3.7 [69 s] (3 unused) | 70.4 ± 0.5 [72 s] (3 unused) | 82.8 [153 s] |

Best stock: 76.6 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## hinglish-top-en

Test rows: 1000 (the same rows for every cell; one standard error is about 1.6 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (285 rows) | 10/opt (519 rows) | 20/opt (884 rows) | 50/opt (1724 rows) | all/opt (4378 rows) |
|---|---|---|---|---|---|---|
| gliner | 77.0 | – | – | – | – | – |
| gliner-multi | 61.3 | – | – | – | – | – |
| djev | 68.6 | – | – | – | – | – |
| djev@3 | 80.2 | – | – | – | – | – |
| gliner+cal | – | 78.8 | 79.1 | 80.0 | – | – |
| tfidf | – | 59.7 ± 3.8 [0 s] | 70.3 ± 0.5 [0 s] | 78.3 ± 0.8 [1 s] (3 unused) | 84.2 ± 0.8 [1 s] (5 unused) | 87.2 ± 0.0 [4 s] (8 unused) |
| tiny | – | 35.5 ± 2.8 [5 s] | 49.6 ± 1.1 [6 s] | 62.6 ± 2.3 [6 s] | 74.7 ± 0.8 [6 s] (5 unused) | 84.9 ± 1.1 [13 s] (4 unused) |
| mmbert | – | 38.7 ± 6.4 [57 s] | 56.0 ± 4.1 [62 s] | 68.8 ± 2.3 [64 s] | 80.2 ± 1.0 [68 s] (3 unused) | 88.8 [153 s] |

Best stock: 80.2 %. A trained model first beats it by more than 2 standard errors at: 50 labels per option.

## civil

Test rows: 1000 (the same rows for every cell; one standard error is about 1.6 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (10 rows) | 10/opt (20 rows) | 20/opt (40 rows) | 50/opt (100 rows) | all/opt (4000 rows) |
|---|---|---|---|---|---|---|
| gliner | 64.8 | – | – | – | – | – |
| djev | 69.6 | – | – | – | – | – |
| djev@3 | 68.2 | – | – | – | – | – |
| gliner+cal | – | 67.9 | 68.0 | 68.1 | – | – |
| djev+cal | – | 71.3 | 71.7 | 72.5 | – | – |
| tfidf | – | 50.7 ± 1.6 [0 s] | 52.2 ± 1.1 [0 s] | 55.5 ± 2.6 [0 s] | 60.6 ± 1.7 [0 s] | 79.1 ± 0.0 [1 s] |
| tiny | – | 53.7 ± 3.9 [1 s] | 54.5 ± 2.2 [2 s] | 60.7 ± 1.8 [4 s] | 65.2 ± 3.4 [12 s] | 80.9 ± 1.2 [70 s] |
| mmbert | – | 55.8 [7 s] | 52.4 [12 s] | 61.5 [26 s] | – | – |

Best stock: 69.6 %. A trained model first beats it by more than 2 standard errors at: all labels per option.

## hinglish-yt

Test rows: 800 (the same rows for every cell; one standard error is about 1.8 pts at 50 %). Cells: accuracy %, mean ± standard deviation over up to 3 seeds (each seed draws different labelled rows and starts training differently); train seconds in brackets; 'k unused' = k test intents the model never predicts. Stage = labelled rows per option (0 = stock, no labels).

| Arm | stock | 5/opt (15 rows) | 10/opt (30 rows) | 20/opt (60 rows) | 50/opt (150 rows) | all/opt (2390 rows) |
|---|---|---|---|---|---|---|
| gliner | 47.1 | – | – | – | – | – |
| gliner-multi | 41.8 | – | – | – | – | – |
| djev | 70.8 | – | – | – | – | – |
| djev@3 | 73.5 | – | – | – | – | – |
| gliner+cal | – | 43.4 | 48.9 | 50.9 | – | – |
| tfidf | – | 41.2 ± 2.7 [0 s] | 42.7 ± 0.3 [0 s] | 45.6 ± 0.8 [0 s] | 52.6 ± 0.6 [0 s] | 67.6 ± 0.0 [1 s] |
| tiny | – | 41.6 ± 1.4 [1 s] | 40.5 ± 3.3 [1 s] | 46.5 ± 4.6 [2 s] | 51.4 ± 1.7 [4 s] | 64.3 ± 1.7 [11 s] |
| mmbert | – | 38.8 ± 7.1 [4 s] | 43.9 ± 4.8 [6 s] | 49.4 ± 2.6 [12 s] | 51.3 ± 2.3 [30 s] | 67.6 [74 s] |

Best stock: 73.5 %. A trained model first beats it by more than 2 standard errors at: no stage measured.

