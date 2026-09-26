# sieve — spec

Tier: **Product** (customer data, a release lifecycle, two roles: the person who labels, the agent that operates).

## §0 Agent contract

Written from the driver's seat (ntkit DRIVER.md v0.1, run 2026-09-25 over the Batch A build). The agent that drives
sieve is context-poor, can be killed mid-command, and pays real money when it touches the djev GPU. Every design
decision below answers to this section.

**The system in one sentence.** Per customer and per question, several models compete on rows the customer
labelled. A candidate goes live only if it beats a fixed bar on rows it never trained on. The live choice is served
exactly as it was scored.

### The tower (each layer reads only the one below)

```
L0 labels      data/customers/<c>/customer.json + data.jsonl      the person's ground truth (person-only writes)
L1 split       djcore.split: sha1(text) % 10: 0-1 select, 2 test, 3-9 train; model-labelled rows train only
L2 reads       backends.py: djev | gliner | tiny -> {option: p}    djev + gliner cached per customer (cache/*.jsonl)
L0' review     djp.review: model answers + held-out rows and least-confident half flagged for a person
L3 candidates  djp.evaluate: base(+cal), profile(+cal), each also @3 (3 reads over shuffled option orders),
               gliner(+cal), tfidf, tiny, mmbert (trained arms need >= 20 rows/option)
L4 gate        djp.choose on select: acc >= bar, log-loss < bar, no more never-predicted options; highest accuracy
               wins (log-loss breaks ties); else the bar. test is reported, never used to choose
L5 release     releases/<v>/release.json + tiny weights; CURRENT; releases/log.jsonl
L6 serve       serve.py -> backends.answer(release): the L5 choice, same reads, same weights
```

An agent enters at the altitude of its task. To see state, it reads L5 (`status`). To add ground truth, a person
works at L0 (`label`). To ask "what would win", it uses L3/L4 (`eval`). To go live, it uses L5 (`release`,
`rollback`). To answer traffic, it uses L6.

### The ten principles, as built

1. **One perception act.** `python3 pipeline/djp.py status --json` gives every customer on one line: the live
   version, holdout accuracy against the bar, the backend per question, and `stale`. `stale` means labels arrived
   after the live release. It uses the stdlib only, makes no model calls and finishes in under 1 s. The server's
   equivalent is `GET /c/<customer>`.
2. **Closed vocabularies.** Verdicts: `CREATED IMPORTED VALID LABELLED RELEASED UNCHANGED ROLLED_BACK GATE_REFUSED
   DATA_INVALID SETUP SERVING`. Backends: `djev gliner tiny`. Candidates: `base base+cal profile profile+cal` (each also
   as `@N`, N = `DJEV_READS`, default 3), `gliner gliner+cal tiny`. Every command ends with one `verdict=<CLASS> key=value …` line. `status --json` and the
   release/log records are JSON.
3. **One verdict per next action.** Exit 0 means proceed. `UNCHANGED` is 0 on purpose: re-running is the right
   move. Exit 2 is `SETUP`: run the command the message names. Exit 7 is `GATE_REFUSED`: add labels or accept the
   bar, and never retry the same data. Exit 8 is `DATA_INVALID`: fix the rows the message lists.
4. **Bounded output.** It grows with customers (`status`) or questions (`eval`, the `release` line). It never grows
   with rows. Per-row detail stays in files: caches, `release.json`, `data/ab/*/results.jsonl`.
5. **Every failure names its remedy.** An unreachable djev names the exact `gcloud run services proxy` command. It
   also offers `--arms gliner,tiny` and says the reads so far are cached. A missing torch says to use
   `.venv/bin/python`. Bad labels list the row, the value and the allowed options.
6. **Crash-safe and idempotent.**
   - `label` rewrites data.jsonl atomically, holding old rows or old + new rows, never a torn append.
   - A release builds in `releases/.tmp-<v>/`, renames into place, then moves `CURRENT` (atomic write).
   - A kill at any point leaves the old release live. The next run clears the scratch dir.
   - A per-customer `flock` keeps label, release and rollback runs from interleaving.
   - Caches are append-only, and a line torn by a crash is skipped.
   - Re-running `label` with the same file, or `release` on the same data and arms, gives `UNCHANGED`.
   - Tests: `tests/test_pipeline.py` (`test_crash_mid_release_keeps_old_live`, `test_release_label_idempotent`).
7. **The tool holds the memory.** `releases/log.jsonl` records every attempt: passed, refused and rollbacks, each
   with data hash, arms, plan and accuracy. Every `release.json` keeps all candidates' holdout metrics.
   `data/tiny/log.jsonl` keeps benchmark runs. `status` renders the trajectory's head.
8. **Accretive by mechanism.**
   - Each `label` run adds ground truth and re-gates, so the holdout grows and the verdicts sharpen.
   - Tiny unlocks by itself for a question once its rarest option reaches 20 labelled rows.
   - The read caches mean each re-release pays only for rows it has not seen.
   - Refused releases are kept, so a candidate that keeps failing shows up as a pattern, not a surprise.
9. **A tower, not a toolbox.** See above. `backends.py` is the only module that knows a model. `djp.py` knows only
   probabilities. `serve.py` knows only releases.
10. **The evaluator stays outside the loop.**
    - The holdout is fixed by hashing the row text. The bar is plain djev averaged over 3 reads in shuffled option
     orders, the model with no customer input, read three times so a single noisy read cannot decide the gate.
    - The gate rule lives in `djp.choose` and is tested.
    - Labels, the gate's ground truth, are **person-only** (`djp.import`, `djp.label` in `tools.json`). An agent
      runs them only on a file the person supplied.
    - It fails closed: invalid data writes nothing, a refused gate leaves `CURRENT` alone, and an unreachable arm
      stops the release instead of dropping the arm.
    - Known weakness, measured: djev's bar varies between reads (up to 5.5 pts on a 72–78 row holdout;
      `results/releases-2026-09-25.md`). The cache freezes one read, so the verdict is reproducible, not certain.
      A gain under about 10 pts at these holdout sizes is noise, and the results file says so per question.

### Doors

Every command and route is declared once in `tools.json`, with its input schema, cost class, whether it mutates, and
`delegable: agent | person-only`. `tests/test_pipeline.py::Manifest` fails if any `djp.py` command
(`djp.COMMANDS`) lacks an entry. The CLI and the HTTP routes are the doors. There is no UI, and so no third door.

### Spend

The djev arm costs $3.19/h while the GPU is up, with a cold start of about 2 minutes. An agent batches djev work
into one window and stops the proxy after it. It asks the person before any run expected to exceed about $2. gliner
and tiny are local and free. `--arms gliner,tiny` keeps a release entirely local, and the bar becomes plain gliner.

## §1 Release record (`releases/<version>/release.json`)

`customer, version, kind="backends", arms, bar, data_sha, djev_reads, base{image, model}, gliner_model, tiny_model, questions,
profile, plan{q: {variant, backend, use_profile, reads, calibration, path?}}, tiny{q: {eligible, least_per_option,
train_s?}}, split{train, holdout, test}, eval{candidate: {q: {accuracy, log_loss, ece, unused}}}, released_eval,
test_eval{n, bar, released}, gate{passed, rule}, seconds`. data.jsonl rows may carry `source: "teacher"` (training
only).

## §2 Open

- djev bar noise: the bar is now averaged over 3 reads (Batch B). Its measured effect is in results/ (Batch B).
- Layer 2 (trained djev weights, TRAINING.md) enters as one more candidate behind the same gate.
