# djev option-order averaging: reads over shuffled option orders (2026-09-26)

**Setup.** Each call was repeated k times. Read 0 shows the options in the order under test. Reads 1 to k-1 show them
in orders shuffled by a seed from (row, order, read). The answer is the mean probability per option across reads, as
in TypeLLM's permutation averaging. The suite is the A/B suite (`results/ab-2026-09-25.md`): 1,700 fast-decisions
rows (2,600 single-choice questions) and the held-out rows of 4 stand-in customers (478 questions). There are 4
orders: original, the same again, reversed and shuffled. djev ran 3 reads on everything and 5 reads on the customer
rows. GLiNER ran 3 reads. Harness: `pipeline/ab.py run --reads K`. Raw rows: `data/ab/perm` and
`data/ab/perm-gliner`, with 0 errors in either.

**Read this first.**
- **3 reads halve djev's order sensitivity on fast-decisions.** Reversing the options flips 23.6 % of answers with 1
  read and 11.8 % with 3. Accuracy rises 1.0 pt (62.7 % → 63.7 %, 2,600 questions, about 1 standard error).
- **On the customer rows, the reversed-order flip rate falls from 31.2 % to 18.8 % at 3 reads and 15.3 % at 5.** The same-order
  noise floor stays at 11–16 %. That noise is djev's own per-call randomness, and more reads average it only slowly. Accuracy
  moves 1–3 pts, within noise on 478 questions.
- **GLiNER gains nothing from averaging.** Accuracy stays within 0.4 pt. Its reversal flips go from 3.0 % to 1.6 %.
- **The bar is steadier, but not fixed.** A fresh second read of the release holdouts changed plain-djev answers on
  10.2 % of rows with 1 read, and on 5.6 % with 3 reads (the mean over 6 questions). The largest accuracy swing stayed large: helpdesk
  · priority moved 5.1 pts at 3 reads (34.6 % → 39.7 %). `data/releases-reread-2026-09-26.json`.
- **Batch A's fragile verdict now holds.** helpdesk · priority's pick (GLiNER + calibration, 46.2 %) beats the averaged
  bar on both reads (34.6 % and 39.7 %). In Batch A, a re-read of a 1-read bar would have failed it.
- **The gate now serves 3 of 6 questions from djev @3:** helpdesk · queue, helpdesk · type and moderation · insult.
  Each costs 3 djev calls per request instead of 1, recorded as `reads: 3` in the release. Only helpdesk · type
  (+14.1 pts, p = 0.007) and shop · category on tiny (+19.4 pts, p = 0.007) are outside noise.
- **Bigger option sets gain far more.** On the Batch C sets (1,000 test rows each), 3 reads took djev
  from 45.3 % to 63.6 % on banking77 (77 options), and from 45.1 % to 69.6 % on MASSIVE English (60). On MASSIVE Hindi (60) it went
  from 38.9 % to 65.0 %, and on CLINC (151) from 52.1 % to 69.6 %. On civil comments (2 options) it went from 69.6 % to 68.2 %. See the Batch C results.

**Cost.** djev ran 23,232 + 1,888 calls at 19.7–20.0 calls/s with 8 parallel callers. That is $0.045 per
1,000 calls while the GPU is busy, so k reads cost k times that. The window ran 23:56–00:23 IST, 26.8 min. With the
cold start and the idle tail, it cost about $2.20 (inferred from uptime × $3.19/h, not billing).

## Accuracy and flips against reads (djev)

| Rows | k | Decisions | Acc orig | Acc again | Acc rev | Acc shuf | Flip again (noise) | Flip rev | Flip shuf |
|---|---|---|---|---|---|---|---|---|---|
| fast-decisions | 1 | 2600 | 62.7% | 62.7% | 59.9% | 61.0% | 10.4% | 23.6% | 19.3% |
| fast-decisions | 2 | 2600 | 62.6% | 63.1% | 63.1% | 63.0% | 10.5% | 14.3% | 12.5% |
| fast-decisions | 3 | 2600 | 63.7% | 64.1% | 63.6% | 63.0% | 8.6% | 11.8% | 10.3% |
| customer | 1 | 478 | 58.8% | 57.3% | 60.7% | 58.8% | 13.0% | 31.2% | 25.7% |
| customer | 2 | 478 | 59.8% | 62.3% | 60.9% | 61.3% | 17.6% | 22.0% | 19.0% |
| customer | 3 | 478 | 59.4% | 60.9% | 60.9% | 61.1% | 15.9% | 18.8% | 17.6% |
| customer | 4 | 478 | 61.5% | 62.1% | 60.9% | 60.5% | 14.0% | 14.2% | 12.6% |
| customer | 5 | 478 | 59.8% | 61.1% | 60.3% | 61.5% | 10.9% | 15.3% | 12.8% |

## GLiNER, for contrast

| Rows | k | Acc orig | Acc rev | Flip again | Flip rev | Flip shuf |
|---|---|---|---|---|---|---|
| fast-decisions | 1 | 67.3% | 67.6% | 0.0% | 3.0% | 1.8% |
| fast-decisions | 3 | 67.2% | 67.5% | 0.8% | 1.6% | 1.3% |
| customer | 1 | 54.4% | 55.0% | 0.0% | 3.8% | 2.3% |
| customer | 3 | 54.2% | 54.2% | 1.3% | 1.7% | 1.9% |

GLiNER's k = 1 row here uses its softmax-over-options mode (`backends.gliner_read`). The 2026-09-25 A/B used
single-label mode. On fast-decisions they agree to 0.0 pt.

## Releases with the averaged bar (B0)

The bar is plain djev averaged over 3 reads (`base@3`). The candidates now include the 1-read and 3-read djev
variants, and the gate rule is unchanged. "Bar re-read" is a fresh, uncached 3-read average of the same holdout.
`compare.py` rebuilt every prediction from caches and saved weights, and asserted each equals the gated
number. It ran with djev unreachable.

| Customer · question | n | Options | Chosen | Chosen acc | Bar acc | Bar re-read | Δ pts | Chosen right / bar right (discordant rows) | Sign test p | base | base+cal | base@3+cal | gliner | gliner+cal | profile | profile+cal | profile@3 | profile@3+cal | tiny |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| helpdesk · queue | 78 | 6 | djev (base@3+cal) | 39.7 | 35.9 | 35.9 | +3.8 | 7 / 4 | 0.549 | 37.2 | 38.5 | 39.7 | 34.6 | 33.3 | 30.8 | 35.9 | 37.2 | 34.6 | 30.8 |
| helpdesk · priority | 78 | 3 | gliner (gliner+cal) | 46.2 | 34.6 | 39.7 | +11.5 | 22 / 13 | 0.175 | 43.6 | 50.0 | 51.3 | 33.3 | 46.2 | 39.7 | 44.9 | 33.3 | 39.7 | 38.5 |
| helpdesk · type | 78 | 4 | djev (profile@3+cal) | 74.4 | 60.3 | 60.3 | +14.1 | 13 / 2 | 0.00739 | 64.1 | 65.4 | 66.7 | 57.7 | 64.1 | 65.4 | 75.6 | 71.8 | 74.4 | – (19/opt) |
| moderation · toxic | 76 | 2 | gliner (gliner+cal) | 81.6 | 73.7 | 73.7 | +7.9 | 13 / 7 | 0.263 | 71.0 | 79.0 | 76.3 | 80.3 | 81.6 | 80.3 | 80.3 | 79.0 | 80.3 | 80.3 |
| moderation · insult | 76 | 2 | djev (profile@3+cal) | 86.8 | 84.2 | 85.5 | +2.6 | 2 / 0 | 0.5 | 85.5 | 85.5 | 84.2 | 60.5 | 82.9 | 79.0 | 77.6 | 86.8 | 86.8 | 77.6 |
| shop · category | 72 | 11 | tiny (tiny) | 90.3 | 70.8 | 70.8 | +19.4 | 19 / 5 | 0.00661 | 70.8 | 66.7 | 68.1 | 59.7 | 61.1 | 70.8 | 69.4 | 72.2 | 69.4 | 90.3 |

Stand-in customers only: shop and helpdesk are synthetic, and moderation is real comments.
