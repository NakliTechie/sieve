# Can zero-shot answers replace human labels? (2026-09-26)

**Question.** Run a zero-shot classifier over unlabelled messages and train a small model on its answers. Is the
student as good as a model trained on human labels, or at least better than its teacher?

**Setup.** Teacher: djev averaged over 3 reads in shuffled option orders (djev@3), zero-shot on the GPU service.
Pools: MASSIVE English and Hindi (up to 100 rows per intent, 5,135 rows) and CLINC150 (30 per intent, 4,530 rows).
Students: Ettin-17M and mmBERT-small, trained on each label set and scored on each set's fixed 1,000 gold test rows
(the rows from `results/stock-vs-trained-2026-09-26.md`). "A person corrects k %" is simulated: the teacher's
least-confident k % of pool rows get their gold label. Harness: `pipeline/distill.py`. Raw:
`data/bench/<set>/distill.jsonl`. GPU service window 08:37–09:21 IST, about $3.10 (inferred from uptime).

**Read this first.**
- **Zero-shot labels alone never beat the teacher.** mmBERT on djev@3's labels is -0.1, -3.4 and -6.1 pts against
  djev@3 itself. The student inherits the teacher's errors: 19–32 % of the pool labels are wrong.
- **With a person correcting the least-confident 25 %, the student beats the teacher on all 3 sets.** The gain is
  +7.2 pts on MASSIVE English and +5.4 on Hindi, both more than 3 standard errors. On CLINC it is +1.4, within noise.
  On MASSIVE that closes 55–65 % of the gap between the unreviewed student and full human labels, for a quarter of
  the labelling.
- **Correct rows; don't drop them.** Training only on the confident half or quarter gives cleaner labels (5–17 %
  error) but fewer and easier rows, and scores lower on every set. Keeping rows where djev and GLiNER agree fails on
  Hindi, where GLiNER cannot read the text (392 rows kept).
- **Label noise is costly.** With 20 % of gold labels flipped at random, mmBERT loses 11–13 pts and Ettin 7–17 pts.
  The noise series (10–40 % flipped) sets the budget for how wrong a teacher's labels can be.
- **The margin shrinks with fewer rows per intent.** CLINC's pool had 30 per intent; its human-label ceiling is
  75.3 %, and the reviewed student sits 4.3 pts below it.
- A 7-set review-budget sweep (0 / 10 / 25 / 50 % least-confident, and a random-25 % control) is running; its
  results land in a follow-up file.

## massive-en

Teacher on the test rows (stock, no labels): djev@3 69.6 %, GLiNER 49.3 %. One standard error at 1,000 rows is about 1.6 pts.

| Pool (per intent) | Label set | Rows | Label error | Ettin-17M | mmBERT-small |
|---|---|---|---|---|---|
| 100 | gold | 5135 | 0.0 % | 78.2 | 82.8 |
| 100 | djev3 | 5135 | 28.8 % | 63.3 | 69.5 |
| 100 | djev3-top50 | 2567 | 13.5 % | 61.3 | 68.4 |
| 100 | djev3-top25 | 1283 | 6.9 % | 47.7 | – |
| 100 | agree | 2458 | 9.5 % | 55.3 | 60.1 |
| 100 | gliner | 5135 | 46.8 % | 46.8 | – |
| 100 | spot10 | 5135 | 20.6 % | 67.9 | – |
| 100 | spot25 | 5135 | 14.3 % | 70.9 | 76.8 |
| 100 | noise10 | 5135 | 10.0 % | 70.8 | – |
| 100 | noise20 | 5135 | 20.0 % | 64.0 | 70.9 |
| 100 | noise30 | 5135 | 30.0 % | 56.8 | – |
| 100 | noise40 | 5135 | 40.0 % | 49.0 | – |

## massive-hi

Teacher on the test rows (stock, no labels): djev@3 65.0 %, GLiNER 6.1 %. One standard error at 1,000 rows is about 1.6 pts.

| Pool (per intent) | Label set | Rows | Label error | Ettin-17M | mmBERT-small |
|---|---|---|---|---|---|
| 100 | gold | 5135 | 0.2 % | 66.7 | 75.1 |
| 100 | djev3 | 5135 | 32.0 % | 54.5 | 61.6 |
| 100 | djev3-top50 | 2567 | 17.0 % | 49.4 | 62.6 |
| 100 | djev3-top25 | 1283 | 9.0 % | 40.1 | – |
| 100 | agree | 392 | 17.6 % | 16.1 | 21.4 |
| 100 | gliner | 5135 | 92.0 % | 10.3 | – |
| 100 | spot10 | 5135 | 24.0 % | 59.0 | – |
| 100 | spot25 | 5135 | 16.9 % | 60.9 | 70.4 |
| 100 | noise10 | 5135 | 10.2 % | 62.1 | – |
| 100 | noise20 | 5135 | 20.2 % | 59.7 | 62.6 |
| 100 | noise30 | 5135 | 30.2 % | 56.7 | – |
| 100 | noise40 | 5135 | 40.1 % | 53.6 | – |

## clinc

Teacher on the test rows (stock, no labels): djev@3 69.6 %, GLiNER 63.9 %. One standard error at 1,000 rows is about 1.6 pts.

| Pool (per intent) | Label set | Rows | Label error | Ettin-17M | mmBERT-small |
|---|---|---|---|---|---|
| 30 | gold | 4530 | 0.0 % | 72.5 | 75.3 |
| 30 | djev3 | 4530 | 19.3 % | 58.7 | 63.5 |
| 30 | djev3-top50 | 2265 | 5.2 % | 51.9 | 56.0 |
| 30 | djev3-top25 | 1132 | 4.5 % | 39.6 | – |
| 30 | agree | 3231 | 6.8 % | 58.0 | 61.9 |
| 30 | gliner | 4530 | 24.9 % | 55.7 | – |
| 30 | spot10 | 4530 | 13.1 % | 63.4 | – |
| 30 | spot25 | 4530 | 8.5 % | 66.6 | 71.0 |
| 30 | noise10 | 4530 | 10.0 % | 63.1 | – |
| 30 | noise20 | 4530 | 20.0 % | 55.4 | 63.9 |
| 30 | noise30 | 4530 | 30.0 % | 48.2 | – |
| 30 | noise40 | 4530 | 40.0 % | 41.4 | – |

Label sets: gold = human labels; djev3 = the teacher's argmax; djev3-top50/top25 = only the most confident half or quarter;
agree = rows where djev3 and GLiNER (multilingual GLiNER for Hindi) agree; gliner = GLiNER as teacher; spot10/25 = djev3 with
the least-confident 10 or 25 % corrected to gold; noise10–40 = gold with that share flipped to a random other intent.
