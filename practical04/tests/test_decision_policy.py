import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_policy import DecisionPolicy, top_k_decisions  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "policy_config.json"


class DecisionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = DecisionPolicy(threshold_low=0.30, threshold_high=0.70)

    def test_below_low_threshold_is_standard(self):
        self.assertEqual(self.policy.decide(0.2999), "standard")

    def test_equal_to_low_threshold_is_manual_review(self):
        self.assertEqual(self.policy.decide(0.30), "manual_review")

    def test_between_thresholds_is_manual_review(self):
        self.assertEqual(self.policy.decide(0.5), "manual_review")

    def test_equal_to_high_threshold_is_priority(self):
        self.assertEqual(self.policy.decide(0.70), "priority_assistance")

    def test_above_high_threshold_is_priority(self):
        self.assertEqual(self.policy.decide(0.95), "priority_assistance")

    def test_score_outside_range_is_rejected(self):
        for bad in (-0.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                self.policy.decide(bad)

    def test_thresholds_in_wrong_order_are_rejected(self):
        with self.assertRaises(ValueError):
            DecisionPolicy(threshold_low=0.8, threshold_high=0.2)

    def test_threshold_outside_range_is_rejected(self):
        with self.assertRaises(ValueError):
            DecisionPolicy(threshold_low=-0.1, threshold_high=0.5)

    def test_batch_keeps_order(self):
        scores = [0.9, 0.1, 0.3, 0.7, 0.5]
        self.assertEqual(
            self.policy.decide_batch(scores),
            ["priority_assistance", "standard", "manual_review", "priority_assistance", "manual_review"],
        )

    def test_single_threshold_policy_has_no_review_zone(self):
        policy = DecisionPolicy(threshold_low=0.4, threshold_high=0.4)
        self.assertEqual(policy.decide_batch([0.39, 0.4, 0.41]), ["standard", "priority_assistance", "priority_assistance"])

    def test_config_file_is_consistent_with_code(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        policy = DecisionPolicy.from_config(CONFIG)
        self.assertEqual(policy.threshold_low, config["thresholds"]["low"])
        self.assertEqual(policy.threshold_high, config["thresholds"]["high"])
        self.assertLessEqual(policy.threshold_low, policy.threshold_high)
        self.assertEqual(policy.decide(config["thresholds"]["high"]), config["actions"][2])

    def test_config_round_trip(self):
        config = {"thresholds": {"low": 0.25, "high": 0.6}, "actions": ["a", "b", "c"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(DecisionPolicy.from_config(path).decide(0.6), "c")


class TopKTest(unittest.TestCase):
    def test_empty_batch(self):
        self.assertEqual(top_k_decisions([], [], 3), [])

    def test_batch_smaller_than_k(self):
        self.assertEqual(top_k_decisions([0.2, 0.9], [1, 2], 5), [True, True])

    def test_exactly_k_selected_in_original_order(self):
        self.assertEqual(top_k_decisions([0.2, 0.9, 0.5, 0.7], [1, 2, 3, 4], 2), [False, True, False, True])

    def test_ties_resolved_by_id(self):
        self.assertEqual(top_k_decisions([0.5, 0.5, 0.5], [30, 10, 20], 2), [False, True, True])

    def test_ties_are_reproducible(self):
        scores, ids = [0.4, 0.8, 0.8, 0.1], ["d", "b", "a", "c"]
        self.assertEqual(top_k_decisions(scores, ids, 1), top_k_decisions(scores, ids, 1))
        self.assertEqual(top_k_decisions(scores, ids, 1), [False, False, True, False])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            top_k_decisions([0.1], [1], -1)
        with self.assertRaises(ValueError):
            top_k_decisions([0.1, 0.2], [1], 1)
        with self.assertRaises(ValueError):
            top_k_decisions([0.1, 0.2], [1, 1], 1)


if __name__ == "__main__":
    unittest.main()
