# Zero-shot labels + a person reviewing the least-confident rows: 9 datasets (2026-09-26)

**Setup.** A zero-shot teacher labels a pool of unlabelled training rows, either djev@3 (3 reads in shuffled option
orders, GPU service) or GLiNER2.5-Decide (local). A person then corrects k % of the rows, simulated with the gold
labels: the rows the teacher is least confident about, or a random k % as the control. mmBERT-small trains on the
result and is scored on each dataset's fixed gold test rows; Ettin-17M trained at 25 % sits in `distill.jsonl`.
Datasets and test rows are the same as in `results/stock-vs-trained-2026-09-26.md` and `results/hinglish-2026-09-26.md`.
"Best zero-shot" is the best no-label model on that test set. Harness: `pipeline/distill.py review`. Raw:
`data/bench/<set>/distill.jsonl`. djev reads came from the cache (GPU window 08:37–09:21 IST); the sweep itself ran
locally 11:20–13:41 IST. One mmBERT training repeated on the same labels moved 1.4 pts (MASSIVE English), so treat
single cells as ±1.5 pts.

**Read this first.**
- **Without review, the student does not beat its teacher (9 of 9).** It ties (+0.6 and +0.2, within noise) or
  trails, by up to 5.5 pts.
- **Reviewing the least-confident rows beats reviewing random rows,** by more than 2 pts on 7 of 9 datasets (2.5–6.8)
  and within noise on the other 2 (civil +0.7, Hinglish sentiment +0.5). The teacher's confidence finds its own
  mistakes: the same number of reviewed rows removes up to twice the label errors.
- **Reviewing half the rows beats the best zero-shot model on 6 of 9 datasets, by 3.5–17.3 pts.** This includes a
  judgement task (civil comments, +9.4) and Hindi (+10.1). On MASSIVE Hindi and CLINC with djev@3 as teacher, the 50 %
  student matches a student trained on all human labels (75.1 vs 75.1 %, 75.4 vs 75.3 %).
- **The 3 failures share one cause: a weak teacher.** Each started with 47–51 % wrong labels (GLiNER on MASSIVE English,
  Hinglish-TOP and Hinglish sentiment). Every success started at 19–34 %. After 50 % review the failures still had
  13–20 % wrong labels; the successes had 3–15 % (the ranges touch at 13–15 %).
- **Rule of thumb from this sweep:** use the best available zero-shot model as the teacher; if its labels are about
  two-thirds right or better, reviewing the least-confident half gives a small model that beats zero-shot. For Indian
  languages that teacher has to be djev-class: GLiNER fails on Hindi and collapses on Hinglish.

| Dataset | Teacher | Teacher zero-shot | Best zero-shot | Label error, no review | mmBERT 0 % | 10 % | 25 % | **50 %** | 25 % random | 50 % vs best zero-shot |
|---|---|---|---|---|---|---|---|---|---|---|
| massive-en | djev@3 | 69.6 | 69.6 (djev@3) | 29 % | 68.1 | 74.2 | 76.3 | **81.0** | 73.8 | +11.4 |
| massive-hi | djev@3 | 65.0 | 65.0 (djev@3) | 32 % | 61.3 | 66.6 | 69.1 | **75.1** | 66.3 | +10.1 |
| clinc | djev@3 | 69.6 | 69.6 (djev@3) | 19 % | 64.1 | 70.1 | 71.9 | **75.4** | 67.4 | +5.8 |
| banking77 | gliner | 70.8 | 70.8 (gliner) | 32 % | 69.9 | 74.8 | 80.3 | **88.1** | 75.3 | +17.3 |
| clinc | gliner | 63.9 | 69.6 (djev@3) | 25 % | 59.8 | 64.3 | 68.7 | **73.1** | 63.8 | +3.5 |
| massive-en | gliner | 49.3 | 69.6 (djev@3) | 47 % | 49.9 | 53.2 | 60.1 | **67.1** | 57.2 | -2.5 |
| civil (judgement) | gliner | 64.8 | 69.6 (djev) | 34 % | 64.7 | 68.0 | 70.5 | **79.0** | 69.8 | +9.4 |
| hinglish-top | gliner | 49.6 | 76.6 (djev@3) | 51 % | 49.8 | 54.2 | 63.4 | **73.3** | 56.6 | -3.3 |
| hinglish-yt (judgement) | gliner | 47.1 | 73.5 (djev@3) | 51 % | 46.5 | 48.2 | 52.1 | **60.4** | 51.6 | -13.1 |

Label error is the share of the teacher's pool labels that disagree with gold, before any review.
