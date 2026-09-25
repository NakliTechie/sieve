"""Three model families behind one interface: rows in, {question: {option: probability}} out.

  djev    /v1/systemone on the GPU service; zero-shot. With a profile, the customer's rules + examples go in front.
  gliner  GLiNER2.5-Decide on this machine; zero-shot. Deterministic.
  tiny    Ettin-17M fine-tuned per question on the customer's training rows; saved inside the release and served
          from there on CPU or MPS (no GPU service).

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
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djcore as P  # noqa: E402

DJEV_MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4@ec4ff3df205028f4e81c954c2227f9312b3ec2ea"
GLINER_MODEL = os.environ.get("GLINER_MODEL", "fastino/GLiNER2.5-Decide")
TINY_MODEL = "jhu-clsp/ettin-encoder-17m"
TINY_MIN_PER_OPTION = 20  # labelled rows per option (all rows, not only training) before tiny is a candidate
TINY_EPOCHS, TINY_LR, TINY_MAX_LEN = 10, 1e-4, 256
ARMS = ("djev", "gliner", "tiny")


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

def djev_read(url, questions, profile, states, cache=None, workers=8):
    """-> [{q: {option: p}}] raw probabilities (no calibration), one /v1/systemone call per uncached state."""
    def one(state):
        text = P.render_state(profile, state)
        k = _key("djev", DJEV_MODEL, questions, text)
        hit = cache.get(k) if cache else None
        if hit is None:
            a = P.ask(url, questions, text)["answers"]
            hit = {q: a[q]["probabilities"] for q in questions}
            if cache:
                cache.put(k, hit)
        return hit
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, states))


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


def tiny_fit(q, spec, train):
    """Fine-tune on the training rows for one question. -> (model, tokenizer, seconds)."""
    T = _tiny()
    return T.train([(r["state"], r["answers"][q]) for r in train], list(spec["criteria"]), TINY_MODEL,
                   TINY_EPOCHS, TINY_LR, TINY_MAX_LEN)


def tiny_proba(model, tok, states):
    return _tiny().proba(model, tok, states, TINY_MAX_LEN)


def tiny_dirname(q):
    return "tiny/" + (re.sub(r"[^A-Za-z0-9_-]", "_", q) or "q") + "-" + hashlib.sha1(q.encode()).hexdigest()[:6]


_tiny_loaded, _tiny_lock = {}, threading.Lock()


def _tiny_model(path):
    with _tiny_lock:
        if path not in _tiny_loaded:
            _tiny_loaded[path] = _tiny().load(path, os.environ.get("SIEVE_DEVICE") or None)
        return _tiny_loaded[path]


# ---------- serving ----------

def answer(url, release, state):
    """Serve one call for a released customer, each question on its planned backend and calibration.
    djev questions: at most two calls (with and without profile), in parallel with the local backends."""
    rdir = P.customer_dir(release["customer"]) / "releases" / release["version"]
    plan, qs = release["plan"], release["questions"]
    t0 = time.time()
    by = {}
    for q, spec in plan.items():
        by.setdefault((spec["backend"], spec.get("use_profile", False)), {})[q] = qs[q]
    raw, calls, model = {}, 0, None

    def djev_group(item):
        (_, use_profile), group = item
        return P.ask(url, group, P.render_state(release["profile"] if use_profile else None, state))

    djev_groups = [i for i in by.items() if i[0][0] == "djev"]
    with ThreadPoolExecutor(max(1, len(djev_groups))) as ex:
        futs = [ex.submit(djev_group, i) for i in djev_groups]
        for (backend, _), group in by.items():
            if backend == "gliner":
                raw.update(gliner_read(group, [state])[0])
                calls += 1
            elif backend == "tiny":
                for q in group:
                    m, tok = _tiny_model(str(rdir / plan[q]["path"]))
                    raw[q] = tiny_proba(m, tok, [state])[0]
                    calls += 1
        for f in futs:
            r = f.result()
            model = r.get("model")
            raw.update({q: a.get("probabilities", {}) for q, a in r.get("answers", {}).items()})
            calls += 1
    out = {}
    for q in plan:
        p = P.calibrate(raw[q], plan[q]["calibration"])
        best = max(p, key=p.get)
        out[q] = {"choice": best, "probabilities": {k: round(v, 6) for k, v in p.items()},
                  "confidence": round(p[best], 6), "backend": plan[q]["backend"]}
    models = {"djev": model or DJEV_MODEL, "gliner": GLINER_MODEL, "tiny": TINY_MODEL}
    return {"model": {b: models[b] for b in sorted({s["backend"] for s in plan.values()})}, "customer": release["customer"], "release": release["version"], "answers": out,
            "diagnostics": {"timing": {"total_ms": round((time.time() - t0) * 1000, 1)}, "calls": calls}}
