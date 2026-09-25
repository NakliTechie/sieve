#!/usr/bin/env python3
"""Serve each customer's live release: POST /c/<customer>/v1/systemone {"state": "..."} -> answers.

Each question is answered on the backend its release chose (backends.answer): tiny from the weights saved in the
release (CPU or MPS; no GPU service), gliner locally, djev through DJEV_URL. CURRENT is re-read on every call, so a
new release or a rollback takes effect on the next request without a restart.

  .venv/bin/python pipeline/serve.py            # then: curl -s localhost:8090/c/shop/v1/systemone -d '{"state": "..."}'

  GET  /healthz                                 -> {"ok": true}
  GET  /c/<customer>                            -> the live release: version, per-question backend, holdout accuracy
  POST /c/<customer>/v1/systemone               -> {"answers": {q: {choice, probabilities, confidence, backend}}, ...}

Env: PORT (8090), HOST (127.0.0.1), DJEV_URL (http://localhost:8081), SIEVE_DEVICE (cpu | mps; default: mps if present).
"""
import json
import os
import sys
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as B  # noqa: E402
import djcore as P  # noqa: E402

PORT = int(os.environ.get("PORT", "8090"))
HOST = os.environ.get("HOST", "127.0.0.1")
URL = os.environ.get("DJEV_URL", "http://localhost:8081")


def err(code, msg):
    return code, {"error": {"message": msg}}


def live(cid):
    if not cid.replace("-", "").replace("_", "").isalnum():
        return None, err(400, "bad customer id")
    r = P.current_release(cid)
    if r is None:
        return None, err(404, f"no live release for '{cid}'. Run: .venv/bin/python pipeline/djp.py release {cid}")
    return r, None


def summary(cid):
    r, e = live(cid)
    if e:
        return e
    return 200, {"customer": cid, "release": r["version"], "bar": r.get("bar", "base"), "holdout": r["split"]["holdout"],
                 "questions": {q: {"backend": p["backend"], "variant": p["variant"], "options": r["questions"][q]["criteria"],
                                   "holdout_accuracy": r["released_eval"][q]["accuracy"]} for q, p in r["plan"].items()}}


def call(cid, body):
    r, e = live(cid)
    if e:
        return e
    try:
        state = json.loads(body or b"{}")["state"]
        assert isinstance(state, str) and state.strip()
    except (ValueError, KeyError, TypeError, AssertionError):
        return err(400, 'body must be {"state": "<non-empty text>"}')
    try:
        return 200, B.answer(URL, r, state)
    except urllib.error.HTTPError as x:
        return 502, {"error": {"message": f"djev answered HTTP {x.code}"}}
    except urllib.error.URLError as x:
        return err(502, f"a question is on djev and {URL} is unreachable ({x.reason}). Start: gcloud run services proxy "
                        "djev --region us-central1 --project djev-ouz56i --port 8081")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parts = self.path.split("?")[0].strip("/").split("/")
        if parts == ["healthz"]:
            return self._send(200, {"ok": True})
        if len(parts) == 2 and parts[0] == "c":
            return self._send(*summary(parts[1]))
        self._send(*err(404, "routes: GET /healthz, GET /c/<customer>, POST /c/<customer>/v1/systemone"))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) == 4 and parts[0] == "c" and parts[2:] == ["v1", "systemone"]:
            return self._send(*call(parts[1], body))
        self._send(*err(404, "routes: GET /healthz, GET /c/<customer>, POST /c/<customer>/v1/systemone"))


if __name__ == "__main__":
    print(f"verdict=SERVING http://{HOST}:{PORT}/c/<customer>/v1/systemone data={P.HOME}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
