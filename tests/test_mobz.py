"""Unit tests for MobZ modules that do not require network access.

Run with:  PYTHONPATH=src python -m pytest -q   (or python -m unittest)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobz.config import Config, ConfigError  # noqa: E402
from mobz.json_loader import load_tasks, TaskLoadError  # noqa: E402
from mobz.cognitive_analyzer import PlaceholderCognitiveAnalyzer  # noqa: E402
from mobz.models import CognitiveProfile, ModelProfile  # noqa: E402
from mobz.policy_engine import InferencePolicyEngine  # noqa: E402
from mobz.profile_store import FileProfileStore  # noqa: E402
from mobz.validator import validate_result, write_results  # noqa: E402
from mobz.models import InferenceResult, TaskResult  # noqa: E402


def _cfg(**over):
    base = dict(
        fireworks_api_key="k",
        fireworks_base_url="http://x",
        allowed_models=["model-a", "model-b"],
    )
    base.update(over)
    return Config(**base)


class TestJsonLoader(unittest.TestCase):
    def _write(self, data):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_array(self):
        path = self._write([{"task_id": "t1", "prompt": "hi"}])
        tasks = list(load_tasks(path))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "t1")

    def test_loads_wrapper_object(self):
        path = self._write({"tasks": [{"task_id": "t1", "prompt": "hi"}]})
        self.assertEqual(len(list(load_tasks(path))), 1)

    def test_rejects_missing_prompt(self):
        path = self._write([{"task_id": "t1"}])
        with self.assertRaises(TaskLoadError):
            load_tasks(path)

    def test_rejects_duplicate_ids(self):
        path = self._write(
            [{"task_id": "t1", "prompt": "a"}, {"task_id": "t1", "prompt": "b"}]
        )
        with self.assertRaises(TaskLoadError):
            load_tasks(path)

    def test_missing_file(self):
        with self.assertRaises(TaskLoadError):
            load_tasks("/nonexistent/tasks.json")


class TestCognitiveAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PlaceholderCognitiveAnalyzer()

    def test_detects_math(self):
        cog = self.analyzer.analyze("Calculate the sum of 2 and 3")
        self.assertEqual(cog.task, "math")
        self.assertTrue(0.0 <= cog.difficulty <= 1.0)

    def test_detects_json_output(self):
        cog = self.analyzer.analyze("Return the answer as JSON with fields a and b")
        self.assertEqual(cog.expected_output, "json")

    def test_always_returns_profile(self):
        cog = self.analyzer.analyze("hello there")
        self.assertIsInstance(cog, CognitiveProfile)


class TestPolicyEngine(unittest.TestCase):
    def setUp(self):
        self.profiles = {
            "model-a": ModelProfile(
                model="model-a",
                capabilities={"math": 0.72, "coding": 0.96},
                avg_tokens=100, latency=500, cost=0.0001, json_reliability=0.9,
            ),
            "model-b": ModelProfile(
                model="model-b",
                capabilities={"math": 0.97, "coding": 0.98},
                avg_tokens=400, latency=1500, cost=0.0009, json_reliability=0.99,
            ),
        }

    def test_prefers_cheaper_model_when_both_pass(self):
        engine = InferencePolicyEngine(_cfg(), self.profiles)
        cog = CognitiveProfile("coding", 0.2, 0.2, "code", 100, 0.9)
        decision = engine.select("t1", cog)
        # Both clear the gate on coding; cheaper/fewer-tokens model-a should win.
        self.assertEqual(decision.selected_model, "model-a")

    def test_hard_task_requires_strong_model(self):
        engine = InferencePolicyEngine(_cfg(quality_threshold=0.7), self.profiles)
        cog = CognitiveProfile("math", 0.95, 0.95, "plain_text", 50, 0.9)
        decision = engine.select("t2", cog)
        # High difficulty pushes the quality bar above model-a's 0.72 math score.
        self.assertEqual(decision.selected_model, "model-b")

    def test_token_budget_from_expected_output(self):
        # max_tokens is a floor-clamped cap; the generous floor prevents
        # reasoning models from truncating (a truncated answer fails the gate).
        engine = InferencePolicyEngine(
            _cfg(output_token_margin=2.0, min_output_tokens=8), self.profiles)
        cog = CognitiveProfile("math", 0.1, 0.1, "plain_text", 10, 0.9)
        decision = engine.select("t3", cog)
        self.assertEqual(decision.max_output_tokens, 20)

    def test_token_budget_respects_floor(self):
        engine = InferencePolicyEngine(_cfg(min_output_tokens=256), self.profiles)
        cog = CognitiveProfile("math", 0.1, 0.1, "plain_text", 10, 0.9)
        decision = engine.select("t3b", cog)
        self.assertEqual(decision.max_output_tokens, 256)


class TestProfileStore(unittest.TestCase):
    def test_reads_array(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        Path(path).write_text(
            json.dumps([{"model": "m1", "math": 0.8, "avg_tokens": 100}]),
            encoding="utf-8",
        )
        profiles = FileProfileStore(path).load_profiles()
        self.assertIn("m1", profiles)
        self.assertAlmostEqual(profiles["m1"].capability("math"), 0.8)

    def test_missing_file_returns_empty(self):
        self.assertEqual(FileProfileStore("/nope.json").load_profiles(), {})


class TestRichProfileStore(unittest.TestCase):
    """Verifies the gold DB is consumable and normalised by slug."""

    DB = str(Path(__file__).resolve().parents[1] / "mobz_model_profiles.json")

    def setUp(self):
        if not Path(self.DB).is_file():
            self.skipTest("gold DB not present")

    def test_loads_and_maps_capabilities(self):
        from mobz.profile_store import RichProfileStore, normalise_id
        profiles = RichProfileStore(self.DB).load_profiles()
        self.assertGreater(len(profiles), 50)
        # Real model resolvable by full API id via slug normalisation.
        api_id = "accounts/fireworks/models/deepseek-v4-pro"
        p = profiles.get(api_id) or profiles.get(normalise_id(api_id))
        self.assertIsNotNone(p)
        self.assertGreater(p.capability("coding"), 0.5)
        self.assertGreater(p.cost, 0.0)   # $/1M output tokens

    def test_normalise_id(self):
        from mobz.profile_store import normalise_id
        self.assertEqual(normalise_id("accounts/fireworks/models/glm-5p2"), "glm-5p2")
        self.assertEqual(normalise_id("fireworks/glm-5p2"), "glm-5p2")
        self.assertEqual(normalise_id("glm-5p2"), "glm-5p2")


class TestPromptCompressor(unittest.TestCase):
    def setUp(self):
        from mobz.prompt_compressor import HeuristicPromptCompressor
        self.c = HeuristicPromptCompressor()

    def test_removes_filler_and_reduces(self):
        p = "Please could you kindly summarise this for me, thank you: sales rose."
        out = self.c.compress(p)
        self.assertLess(len(out.split()), len(p.split()))
        self.assertNotIn("please", out.lower())

    def test_preserves_numbers(self):
        from mobz.prompt_compressor import is_safe_compression
        p = "Please compute 47 times 23 and 15% of 200."
        out = self.c.compress(p)
        self.assertTrue(is_safe_compression(p, out))
        self.assertIn("47", out)
        self.assertIn("23", out)

    def test_keeps_original_when_no_gain(self):
        p = "What is 2 plus 2?"
        self.assertEqual(self.c.compress(p), p)

    def test_dedupes_repeated_instruction(self):
        p = "Write a function. Write a function."
        out = self.c.compress(p)
        self.assertLess(len(out.split()), len(p.split()))

    def test_unsafe_compression_rejected(self):
        from mobz.prompt_compressor import is_safe_compression
        self.assertFalse(is_safe_compression("Compute 42 now", "Compute now"))  # lost 42
        self.assertFalse(is_safe_compression("hello world", ""))                # empty

    def test_noop_compressor(self):
        from mobz.prompt_compressor import NoopPromptCompressor
        p = "unchanged prompt 123"
        self.assertEqual(NoopPromptCompressor().compress(p), p)


class TestValidator(unittest.TestCase):
    def test_empty_answer_gets_fallback(self):
        res = InferenceResult(task_id="t1", model="m", answer="")
        out = validate_result(res)
        self.assertEqual(out.answer, "N/A")

    def test_json_extraction(self):
        res = InferenceResult(task_id="t1", model="m", answer='Here: {"a": 1} done')
        cog = CognitiveProfile("ner", 0.2, 0.2, "json", 50, 0.9)
        out = validate_result(res, cog)
        self.assertEqual(json.loads(out.answer), {"a": 1})

    def test_write_results_is_valid_json(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        n = write_results(path, [TaskResult("t1", "hello")])
        self.assertEqual(n, 1)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data, [{"task_id": "t1", "answer": "hello"}])


class TestConfig(unittest.TestCase):
    def test_missing_required_raises(self):
        env_backup = dict(os.environ)
        for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
            os.environ.pop(key, None)
        try:
            with self.assertRaises(ConfigError):
                Config.from_env()
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


if __name__ == "__main__":
    unittest.main(verbosity=2)
