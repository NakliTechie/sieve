"""Release pipeline: the gate, label idempotence, crash safety, gate refusal, manifest parity.

  python3 -m unittest discover tests        # stdlib only; the model arms are replaced by fakes, no torch, no network
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="sieve-test-"))
os.environ["DJP_HOME"] = str(TMP / "home")
os.environ["SIEVE_DATA"] = str(TMP / "data")
sys.path.insert(0, str(ROOT / "pipeline"))
import backends as B  # noqa: E402
import djp  # noqa: E402

OPTIONS = ["red", "blue"]


def rows(n, start=0):
    return [{"state": f"ticket {i}: the item is {OPTIONS[i % 2]}", "answers": {"color": OPTIONS[i % 2]}}
            for i in range(start, start + n)]


class FakeTiny:
    """Reads the colour from the text: right on every row (or wrong, when `wrong`)."""
    wrong = False

    @staticmethod
    def save(model, tok, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.safetensors").write_text("fake")


def fake_tiny_proba(model, tok, states):
    out = []
    for s in states:
        hit = "red" if "red" in s else "blue"
        if FakeTiny.wrong:
            hit = "blue" if hit == "red" else "red"
        out.append({o: 0.9 if o == hit else 0.1 for o in OPTIONS})
    return out


def fake_gliner_read(questions, states, cache=None):
    return [{q: {o: 1 / len(s["criteria"]) for o in s["criteria"]} for q, s in questions.items()} for _ in states]


class Pipeline(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP / "home", ignore_errors=True)
        self.d = Path(os.environ["DJP_HOME"]) / "acme"
        self.d.mkdir(parents=True)
        (self.d / "customer.json").write_text(json.dumps({
            "id": "acme", "questions": {"color": {"type": "choice", "instructions": "Colour of the item.", "criteria": OPTIONS}},
            "context": "Colours.", "few_shot": 2}))
        (self.d / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(120)))
        self._saved = (B.gliner_read, B.tiny_fit, B.tiny_proba, B._tiny, os.replace)
        B.gliner_read = fake_gliner_read
        B.tiny_fit = lambda q, spec, train: (None, None, 0.1)
        B.tiny_proba = fake_tiny_proba
        B._tiny = lambda: FakeTiny
        FakeTiny.wrong = False

    def tearDown(self):
        B.gliner_read, B.tiny_fit, B.tiny_proba, B._tiny, os.replace = self._saved

    def release(self, *args):
        """-> exit code of djp.main."""
        try:
            return djp.main(list(args))
        except SystemExit as e:
            return e.code

    def current(self):
        cur = self.d / "releases" / "CURRENT"
        return cur.read_text().strip() if cur.exists() else None

    def test_gate_picks_lowest_logloss_among_passing(self):
        ev = {"base": {"a": {"accuracy": 0.6, "log_loss": 1.0}, "b": {"accuracy": 0.8, "log_loss": 0.5}},
              "tiny": {"a": {"accuracy": 0.9, "log_loss": 0.4}},
              "gliner": {"a": {"accuracy": 0.6, "log_loss": 0.3}, "b": {"accuracy": 0.7, "log_loss": 0.1}},
              "base+cal": {"a": {"accuracy": 0.5, "log_loss": 0.2}, "b": {"accuracy": 0.8, "log_loss": 0.6}}}
        # a: base+cal loses accuracy; gliner ties the bar on accuracy with lower log-loss than tiny -> gliner
        # b: gliner loses accuracy, base+cal has higher log-loss, tiny does not cover b -> stays on the bar
        self.assertEqual(djp.choose(["a", "b"], ev, "base"), {"a": "gliner", "b": "base"})

    def test_release_label_idempotent(self):
        self.assertEqual(self.release("release", "acme", "--arms", "gliner,tiny"), 0)
        first = self.current()
        rel = json.loads((self.d / "releases" / first / "release.json").read_text())
        self.assertEqual(rel["plan"]["color"]["backend"], "tiny")
        self.assertTrue((self.d / "releases" / first / rel["plan"]["color"]["path"] / "model.safetensors").exists())
        self.assertEqual(self.release("release", "acme", "--arms", "gliner,tiny"), 0)
        self.assertEqual(self.current(), first, "same data + arms must not cut a new release")
        new = TMP / "new.jsonl"
        new.write_text("".join(json.dumps(r) + "\n" for r in rows(20, start=500)))
        self.assertEqual(self.release("label", "acme", str(new)), 0)
        second = self.current()
        self.assertNotEqual(second, first)
        self.assertEqual(len((self.d / "data.jsonl").read_text().splitlines()), 140)
        self.assertEqual(self.release("label", "acme", str(new)), 0)
        self.assertEqual(self.current(), second)
        self.assertEqual(len((self.d / "data.jsonl").read_text().splitlines()), 140)
        log = [json.loads(l) for l in (self.d / "releases" / "log.jsonl").read_text().splitlines()]
        self.assertEqual([x["version"] for x in log], [first, second])

    def test_label_rejects_bad_rows_and_writes_nothing(self):
        before = (self.d / "data.jsonl").read_text()
        bad = TMP / "bad.jsonl"
        conflict = dict(rows(1)[0], answers={"color": "blue"})  # row 0 is red in data.jsonl
        bad.write_text(json.dumps({"state": "x", "answers": {"color": "green"}}) + "\n" + json.dumps(conflict) + "\n")
        self.assertEqual(self.release("label", "acme", str(bad)), 8)
        self.assertEqual((self.d / "data.jsonl").read_text(), before)
        self.assertIsNone(self.current())

    def test_crash_mid_release_keeps_old_live(self):
        self.assertEqual(self.release("release", "acme", "--arms", "gliner,tiny"), 0)
        first = self.current()
        (self.d / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(160)))
        real = os.replace

        def boom(src, dst):
            if Path(src).name.startswith(".tmp-"):
                raise KeyboardInterrupt("killed mid-release")
            return real(src, dst)
        os.replace = boom
        with self.assertRaises(KeyboardInterrupt):
            djp.main(["release", "acme", "--arms", "gliner,tiny"])
        os.replace = real
        self.assertEqual(self.current(), first)
        self.assertTrue(list((self.d / "releases").glob(".tmp-*")))
        self.assertEqual(self.release("release", "acme", "--arms", "gliner,tiny"), 0)
        self.assertNotEqual(self.current(), first)
        self.assertFalse(list((self.d / "releases").glob(".tmp-*")), "the next release clears the scratch dir")

    def test_gate_refusal_leaves_current(self):
        FakeTiny.wrong = True
        self.assertEqual(self.release("release", "acme", "--arms", "gliner,tiny"), 7)
        self.assertIsNone(self.current())
        self.assertEqual(len(list((self.d / "releases").glob("2*"))), 1, "the refused record is kept")

    def test_rollback(self):
        self.assertEqual(self.release("rollback", "acme"), 2, "nothing to roll back to")
        self.assertEqual(self.release("release", "acme", "--arms", "gliner,tiny"), 0)
        first = self.current()
        self.assertEqual(self.release("release", "acme", "--arms", "gliner,tiny", "--force"), 0)
        self.assertNotEqual(self.current(), first)
        self.assertEqual(self.release("rollback", "acme"), 0)
        self.assertEqual(self.current(), first)

    def test_tiny_needs_twenty_per_option(self):
        few = rows(30)
        ok, least = B.tiny_eligible({"criteria": OPTIONS}, "color", few)
        self.assertEqual((ok, least), (False, 15))
        self.assertTrue(B.tiny_eligible({"criteria": OPTIONS}, "color", rows(40))[0])


def fake_ask(url, questions, state, **kw):
    """A djev stand-in with a position bias: 0.6 on the first option shown, the rest shared."""
    out = {}
    for q, spec in questions.items():
        opts = spec["criteria"]
        out[q] = {"probabilities": {o: 0.6 if j == 0 else 0.4 / (len(opts) - 1) for j, o in enumerate(opts)}}
    return {"answers": out, "model": "fake"}


class Averaging(Pipeline):
    def setUp(self):
        super().setUp()
        self._ask = djp.P.ask
        djp.P.ask = fake_ask

    def tearDown(self):
        djp.P.ask = self._ask
        super().tearDown()

    def test_permute_is_deterministic_and_read0_is_identity(self):
        qs = {"q": {"criteria": ["a", "b", "c", "d"]}}
        self.assertIs(B.permute(qs, "s", 0), qs)
        self.assertEqual(B.permute(qs, "s", 1), B.permute(qs, "s", 1))
        self.assertEqual(sorted(B.permute(qs, "s", 2)["q"]["criteria"]), ["a", "b", "c", "d"])

    def test_averaging_dilutes_position_bias(self):
        qs = {"q": {"criteria": ["a", "b", "c", "d"]}}
        one = B.djev_read("x", qs, None, ["some text"], None, 1, 1)[0]["q"]
        many = B.djev_read("x", qs, None, ["some text"], None, 1, 8)[0]["q"]
        self.assertAlmostEqual(one["a"], 0.6)
        self.assertLess(max(many.values()), 0.6)
        self.assertAlmostEqual(sum(many.values()), 1.0)

    def test_djev_candidates_and_averaged_bar(self):
        self.assertEqual(djp.candidate("profile@3+cal"), ("djev", True, 3))
        self.assertEqual(djp.candidate("base"), ("djev", False, 1))
        self.assertEqual(djp.candidate("gliner+cal"), ("gliner", False, 1))
        self.assertEqual(self.release("release", "acme", "--arms", "djev,tiny"), 0)
        rel = json.loads((self.d / "releases" / self.current() / "release.json").read_text())
        self.assertEqual(rel["bar"], "base@3")
        self.assertIn("profile@3+cal", rel["eval"])
        self.assertEqual(rel["djev_reads"], 3)
        self.assertEqual(rel["plan"]["color"]["backend"], "tiny")


    def test_serving_averages_the_same_reads_as_evaluation(self):
        qs = {"color": {"criteria": ["red", "blue", "green"]}}
        release = {"customer": "acme", "version": "v", "questions": qs, "profile": None,
                   "plan": {"color": {"backend": "djev", "use_profile": False, "reads": 3, "calibration": {}}}}
        served = B.answer("x", release, "a green thing")
        evald = B.djev_read("x", qs, None, ["a green thing"], None, 1, 3)[0]["color"]
        self.assertEqual(served["diagnostics"]["calls"], 3)
        for o, p in evald.items():
            self.assertAlmostEqual(served["answers"]["color"]["probabilities"][o], p, places=5)


class Manifest(unittest.TestCase):
    def test_manifest_covers_every_command(self):
        m = json.loads((ROOT / "tools.json").read_text())
        names = {t["name"] for t in m["tools"]}
        for cmd in djp.COMMANDS:
            self.assertIn(f"djp.{cmd}", names)
        for extra in ("serve.answer", "serve.release", "samples.build", "ab.run", "ab.report", "tiny.banking77",
                      "tiny.customer", "djp.rollback"):
            self.assertIn(extra, names)
        for t in m["tools"]:
            self.assertIn(t.get("delegable"), ("agent", "person-only"), t["name"])
            self.assertIn("inputSchema", t, t["name"])
            self.assertIn("run", t, t["name"])


if __name__ == "__main__":
    unittest.main()
