"""Three model families behind one interface: rows in, {question: {option: probability}} out.

  djev    /v1/systemone on the GPU service; zero-shot. With a profile, the customer's rules + examples go in front.
  gliner  GLiNER2.5-Decide on this machine; zero-shot. Deterministic.
  tiny    Ettin-17M fine-tuned per question on the customer's training rows; saved inside the release and served
          from there on CPU or MPS (no GPU service).
  mmbert  mmBERT-small (140M, multilingual) fine-tuned and served the same way; +7-12 pts over tiny on Hindi
          (results/stock-vs-trained-2026-09-26.md). 128 tokens: at 256 it grew past 11 GB on MPS.

The release pipeline (djp.py) and the server (serve.py) both call this module, so what the gate evaluated is what is
served: djev and gliner are the same calls, tiny is the same saved weights.

Reads are cached per customer in <customer>/cache/<arm>.jsonl (append-only; a torn last line is skipped), keyed by the
model, the questions and the exact text sent. A re-release after new labels only pays for rows it has not seen. djev
answers vary between calls; the cache freezes one read per row, so the gate is reproducible.

gliner and tiny need torch + transformers + gliner2 (the .venv); they are imported only when used.
"""
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djcore as P  # noqa: E402

DJEV_MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4@ec4ff3df205028f4e81c954c2227f9312b3ec2ea"
GLINER_MODEL = os.environ.get("GLINER_MODEL", "fastino/GLiNER2.5-Decide")
TINY_MIN_PER_OPTION = 20  # labelled rows per option (all rows, not only training) before tiny is a candidate
TINY_EPOCHS = 10  # both trained arms
DJEV_READS = int(os.environ.get("DJEV_READS", "3"))  # reads averaged over option orders for the bar and the @N candidates
ARMS = ("djev", "gliner", "tiny", "mmbert")
# trained arms: model, learning rate, max tokens
TRAINED = {"tiny": ("jhu-clsp/ettin-encoder-17m", 1e-4, 256), "mmbert": ("jhu-clsp/mmBERT-small", 5e-5, 128)}


class NeedsVenv(RuntimeError):
    """An arm needs torch/gliner2 and this interpreter lacks them."""


def _need_torch(arm):
    """The model arms need torch >= 2.6 (transformers refuses older torch for these checkpoints): the .venv has it."""
    try:
        import torch
    except ImportError as e:
        raise NeedsVenv(f"{arm} arm needs torch ({e})") from e
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        raise NeedsVenv(f"{arm} arm needs torch >= 2.6; {sys.executable} has {torch.__version__}")


# ---------- cache ----------

class Cache:
    def __init__(self, path):
        self.path, self.d, self.lock = path, {}, threading.Lock()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    k, v = json.loads(line)
                    self.d[k] = v
                except ValueError:
                    pass  # a line torn by a crash mid-write; the read is redone

    def get(self, key):
        return self.d.get(key)

    def put(self, key, value):
        with self.lock:
            self.d[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps([key, value]) + "\n")


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def cache_for(cid, arm):
    return Cache(P.customer_dir(cid) / "cache" / f"{arm}.jsonl")


# ---------- djev ----------

def permute(questions, state, i):
    """Read i of a state: read 0 shows the options as given; read i > 0 shows each question's options in an order
    shuffled by a seed from (state, question, i). Same text -> same orders, so evaluation and serving see the same
    reads. Averaging reads over orders is the published fix for position effects (TypeLLM permutation averaging)."""
    if i == 0:
        return questions
    out = {}
    for q, spec in questions.items():
        opts = list(spec["criteria"])
        random.Random(_key("perm", state, q, i)).shuffle(opts)
        out[q] = {**spec, "criteria": opts}
    return out


def average(reads):
    """[{q: {option: p}}] over reads of one state -> {q: {option: mean p}}."""
    return {q: {o: sum(r[q].get(o, 0.0) for r in reads) / len(reads) for o in reads[0][q]} for q in reads[0]}


def djev_read(url, questions, profile, states, cache=None, workers=8, reads=1):
    """-> [{q: {option: p}}] raw probabilities (no calibration), averaged over `reads` option orders (permute);
    one /v1/systemone call per uncached (state, read)."""
    def one(job):
        state, i = job
        text = P.render_state(profile, state)
        qs = permute(questions, state, i)
        k = _key("djev", DJEV_MODEL, qs, text) if i == 0 else _key("djev", DJEV_MODEL, qs, text, i)
        hit = cache.get(k) if cache else None
        if hit is None:
            a = P.ask(url, qs, text)["answers"]
            hit = {q: a[q]["probabilities"] for q in qs}
            if cache:
                cache.put(k, hit)
        return hit
    jobs = [(s, i) for s in states for i in range(reads)]
    with ThreadPoolExecutor(workers) as ex:
        flat = list(ex.map(one, jobs))
    return [average(flat[j * reads:(j + 1) * reads]) for j in range(len(states))]


# ---------- gliner ----------

_gl, _gl_lock = None, threading.Lock()


def _gliner():
    global _gl
    if _gl is None:
        _need_torch("gliner")
        try:
            from gliner2 import AutoExtractor
        except ImportError as e:
            raise NeedsVenv(f"gliner arm needs gliner2 ({e})") from e
        _gl = AutoExtractor.from_pretrained(GLINER_MODEL)
    return _gl


def gliner_read(questions, states, cache=None):
    """-> [{q: {option: p}}]. The same information the A/B gave GLiNER: question name + options.
    All options come back (softmax over them), so log-loss and calibration apply as for djev."""
    opts = {q: list(s["criteria"]) for q, s in questions.items()}
    out = []
    for state in states:
        k = _key("gliner", GLINER_MODEL, opts, state)
        hit = cache.get(k) if cache else None
        if hit is None:
            with _gl_lock:  # one model instance; GPU/MPS calls serialised
                m = _gliner()
                schema = m.create_schema()
                for q, o in opts.items():
                    schema = schema.classification(q, o, multi_label=True, cls_threshold=0.0, class_act="softmax")
                r = m.extract(state, schema, include_confidence=True)
            hit = {q: _gliner_probs(r.get(q), opts[q]) for q in opts}
            if cache:
                cache.put(k, hit)
        out.append(hit)
    return out


def _gliner_probs(res, options):
    """gliner2 multi-label output with threshold 0 -> {option: p} over every option (missing ones get 0)."""
    items = res if isinstance(res, list) else [res] if res else []
    p = {o: 0.0 for o in options}
    for it in items:
        if isinstance(it, dict):
            label, conf = it.get("label"), it.get("confidence")
        else:
            label, conf = it[0], it[1]
        if label in p:
            p[label] = float(conf)
    z = sum(p.values()) or 1.0
    return {o: v / z for o, v in p.items()}


# ---------- tiny ----------

def _tiny():
    _need_torch("tiny")
    try:
        import tiny
    except ImportError as e:
        raise NeedsVenv(f"tiny arm needs torch + transformers ({e})") from e
    return tiny


def tiny_eligible(question_spec, q, rows):
    """-> (eligible, fewest labelled rows of any option)."""
    counts = {o: 0 for o in question_spec["criteria"]}
    for r in rows:
        if r["answers"].get(q) in counts:
            counts[r["answers"][q]] += 1
    least = min(counts.values())
    return least >= TINY_MIN_PER_OPTION, least


def tiny_fit(q, spec, train, arm="tiny"):
    """Fine-tune on the training rows for one question. -> (model, tokenizer, seconds).
    One training at a time per machine (a lock in $SIEVE_DATA): two mmBERT runs overlapping on one Mac's GPU
    (17.7 GB together) panicked the kernel on 2026-09-26."""
    import fcntl
    T = _tiny()
    model_id, lr, max_len = TRAINED[arm]
    P.DATA.mkdir(parents=True, exist_ok=True)
    with open(P.DATA / ".train.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return T.train([(r["state"], r["answers"][q]) for r in train], list(spec["criteria"]), model_id,
                       TINY_EPOCHS, lr, max_len)


def park(model):
    """Move a trained model to CPU and return the accelerator's cache (a release trains up to 2 x questions models;
    held on MPS they reached 13 GB on helpdesk)."""
    if not hasattr(model, "to"):  # a stand-in model (tests)
        return model
    import gc
    import torch
    model = model.to("cpu")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return model


def tiny_proba(model, tok, states, arm="tiny"):
    return _tiny().proba(model, tok, states, TRAINED[arm][2], 32 if arm == "mmbert" else 128)


def tiny_dirname(q, arm="tiny"):
    return f"{arm}/" + (re.sub(r"[^A-Za-z0-9_-]", "_", q) or "q") + "-" + hashlib.sha1(q.encode()).hexdigest()[:6]


_tiny_loaded, _tiny_lock = {}, threading.Lock()


def _tiny_model(path):
    with _tiny_lock:
        if path not in _tiny_loaded:
            _tiny_loaded[path] = _tiny().load(path, os.environ.get("SIEVE_DEVICE") or None)
        return _tiny_loaded[path]


# ---------- serving ----------

def answer(url, release, state):
    """Serve one call for a released customer, each question on its planned backend and calibration.
    djev questions: one call per (with/without profile) x read, all in parallel with the local backends."""
    rdir = P.customer_dir(release["customer"]) / "releases" / release["version"]
    plan, qs = release["plan"], release["questions"]
    t0 = time.time()
    by = {}
    for q, spec in plan.items():
        by.setdefault((spec["backend"], spec.get("use_profile", False), spec.get("reads", 1)), {})[q] = qs[q]
    raw, calls, model = {}, 0, None

    def djev_call(job):
        (_, use_profile, _), group, i = job
        return P.ask(url, permute(group, state, i), P.render_state(release["profile"] if use_profile else None, state))

    djev_jobs = [(k, g, i) for k, g in by.items() if k[0] == "djev" for i in range(k[2])]
    with ThreadPoolExecutor(max(1, len(djev_jobs))) as ex:
        futs = [(job, ex.submit(djev_call, job)) for job in djev_jobs]
        for (backend, _, _), group in by.items():
            if backend == "gliner":
                raw.update(gliner_read(group, [state])[0])
                calls += 1
            elif backend in TRAINED:
                for q in group:
                    m, tok = _tiny_model(str(rdir / plan[q]["path"]))
                    raw[q] = tiny_proba(m, tok, [state], backend)[0]
                    calls += 1
        per_group = {}
        for (key, group, i), f in futs:
            r = f.result()
            model = r.get("model")
            per_group.setdefault(key, []).append({q: a.get("probabilities", {}) for q, a in r.get("answers", {}).items()})
            calls += 1
        for reads in per_group.values():
            raw.update(average(reads))
    out = {}
    for q in plan:
        p = P.calibrate(raw[q], plan[q]["calibration"])
        best = max(p, key=p.get)
        out[q] = {"choice": best, "probabilities": {k: round(v, 6) for k, v in p.items()},
                  "confidence": round(p[best], 6), "backend": plan[q]["backend"]}
    models = {"djev": model or DJEV_MODEL, "gliner": GLINER_MODEL, **{a: m for a, (m, _, _) in TRAINED.items()}}
    return {"model": {b: models[b] for b in sorted({s["backend"] for s in plan.values()})}, "customer": release["customer"], "release": release["version"], "answers": out,
            "diagnostics": {"timing": {"total_ms": round((time.time() - t0) * 1000, 1)}, "calls": calls}}
