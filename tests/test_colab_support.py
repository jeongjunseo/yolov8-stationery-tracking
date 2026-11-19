"""Mocked Colab batch-policy checks; no Torch import or training."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.colab_support import enforce_explicit_batch_policy, require_supported_ultralytics


class ExplicitBatchPolicyTests(unittest.TestCase):
    def test_each_permitted_batch_disables_internal_retry(self):
        for batch in (16, 8, 4):
            with self.subTest(batch=batch):
                trainer = SimpleNamespace(batch_size=batch, _oom_retries=0)
                enforce_explicit_batch_policy(trainer, batch)
                # This is the retry condition in the pinned trainer. Even an
                # OOM is re-raised for the notebook to decide what to do next.
                self.assertFalse(trainer._oom_retries < 3)
                self.assertEqual(trainer.batch_size, batch)

    def test_batch_mismatch_or_batch_below_four_is_rejected(self):
        for requested, actual in ((16, 8), (4, 2), (2, 2)):
            with self.subTest(requested=requested, actual=actual):
                trainer = SimpleNamespace(batch_size=actual, _oom_retries=0)
                with self.assertRaisesRegex(RuntimeError, "Unexpected training batch"):
                    enforce_explicit_batch_policy(trainer, requested)

    def test_missing_trainer_contract_stops_before_training(self):
        with self.assertRaisesRegex(RuntimeError, "retry state is missing"):
            enforce_explicit_batch_policy(SimpleNamespace(batch_size=16), 16)

    def test_version_guard_accepts_only_reviewed_pin(self):
        with patch("scripts.colab_support.version", return_value="8.4.215"):
            require_supported_ultralytics()
        with patch("scripts.colab_support.version", return_value="8.4.216"):
            with self.assertRaisesRegex(RuntimeError, "Review the trainer callback contract"):
                require_supported_ultralytics()


if __name__ == "__main__":
    unittest.main()
