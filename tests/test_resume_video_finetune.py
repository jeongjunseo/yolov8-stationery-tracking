"""Recovery guards and callbacks with temporary files/fake trainers; no GPU run."""
import copy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import resume_video_finetune as resume


class FakeEMA:
    names = {0: "highlighter", 1: "correction_tape", 2: "eraser"}
    def float(self):
        return self
    def state_dict(self):
        return {"weight": [1.0, 2.0], "count": 3}


def history():
    return [{"epoch": str(i), "time": str(resume.SOURCE_SECONDS if i == 97 else i * 90),
             "lr/pg0": "7.08784e-05"} for i in range(1, 98)]


def checkpoint():
    rows = history()
    args = {"epochs": 100, "batch": 16, "imgsz": 640, "seed": 42, "deterministic": True,
            "workers": 4, "optimizer": "AdamW", "lr0": 0.001429, "lrf": 0.01,
            "cos_lr": False, "warmup_bias_lr": 0.0, "close_mosaic": 10, "amp": True,
            "save_dir": "/content/old/runs/train_video_finetune/stationery_video_b16"}
    groups = [{"params": [i], "lr": 7.08784e-05, "initial_lr": 0.001429,
               "betas": (0.9, 0.999), "weight_decay": 0.0005, "fused": True} for i in range(3)]
    return {"epoch": 96, "version": "8.4.215", "train_args": args,
            "optimizer": {"state": {i: {"step": 873, "exp_avg": [0.1], "exp_avg_sq": [0.2]} for i in range(3)},
                          "param_groups": groups}, "ema": FakeEMA(), "updates": 873,
            "scaler": {"scale": 512.0, "_growth_tracker": 712},
            "best_fitness": 0.83085, "stopper": {"best_epoch": 82, "best_fitness": 0.83085},
            "train_results": {key: [float(row[key]) for row in rows] for key in rows[0]}}


def trainer_for(ckpt, save_dir):
    optimizer = copy.deepcopy(ckpt["optimizer"])
    for group in optimizer["param_groups"]:
        group["fused"] = False  # This runtime-only choice is explicitly permitted.
    return SimpleNamespace(device=SimpleNamespace(type="cuda"), start_epoch=97, epochs=100,
        batch_size=16, save_dir=save_dir, args=SimpleNamespace(**ckpt["train_args"]),
        optimizer=SimpleNamespace(state_dict=lambda: optimizer),
        scaler=SimpleNamespace(state_dict=lambda: copy.deepcopy(ckpt["scaler"])),
        ema=SimpleNamespace(updates=873, ema=FakeEMA()), best_fitness=0.83085,
        scheduler=SimpleNamespace(last_epoch=96, base_lrs=[0.001429] * 3),
        stopper=SimpleNamespace(**ckpt["stopper"]), train_time_start=10000.0)


class RecoveryTests(unittest.TestCase):
    def test_unstripped_checkpoint_history_and_resolved_settings_are_required(self):
        good = checkpoint()
        self.assertEqual(resume.validate_checkpoint(good, history())["completed_epochs"], 97)
        for mutate in (lambda c: c.update(epoch=-1), lambda c: c.update(optimizer=None),
                       lambda c: c["train_args"].update(optimizer="auto", lr0=0.01),
                       lambda c: c["train_results"]["time"].__setitem__(-1, 10.0)):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaises(ValueError):
                resume.validate_checkpoint(bad, history())

    def test_runtime_state_restoration_and_cumulative_time_offset(self):
        ckpt = checkpoint()
        path = Path(tempfile.gettempdir()).resolve() / "resume_fixture_run"
        trainer = trainer_for(ckpt, path)
        evidence = resume.validate_runtime_start(trainer, ckpt, {"recovery_run_directory": str(path)})
        self.assertAlmostEqual(trainer.train_time_start, 10000 - 9447.17)
        self.assertEqual(evidence["next_epoch_one_based"], 98)
        self.assertTrue(evidence["optimizer_state_matches"])
        self.assertEqual(ckpt["optimizer"]["param_groups"][0]["fused"], True)

    def test_wrong_runtime_epoch_optimizer_or_scheduler_stops_before_time_change(self):
        ckpt = checkpoint()
        path = Path(tempfile.gettempdir()).resolve() / "resume_fixture_run"
        for change in (lambda t: setattr(t, "start_epoch", 0),
                       lambda t: t.optimizer.state_dict()["state"][0].update(step=1),
                       lambda t: setattr(t.scheduler, "last_epoch", 0),
                       lambda t: setattr(t, "batch_size", 8)):
            trainer = trainer_for(ckpt, path)
            change(trainer)
            with self.assertRaises(RuntimeError):
                resume.validate_runtime_start(trainer, ckpt, {"recovery_run_directory": str(path)})
            self.assertEqual(trainer.train_time_start, 10000.0)

    def test_csv_must_have_contiguous_epochs_and_increasing_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            rows = history()
            def write():
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            self.assertEqual(len(resume.read_history(path, 97)), 97)
            rows[-1]["epoch"] = "96"
            write()
            with self.assertRaises(ValueError):
                resume.read_history(path, 97)
            rows[-1]["epoch"], rows[-1]["time"] = "97", "1"
            write()
            with self.assertRaises(ValueError):
                resume.read_history(path, 97)

    def test_existing_drive_recovery_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "stationery_video_finetune_newid"
            backup = base / "colab_runs/newid"
            root.mkdir()
            backup.mkdir(parents=True)
            sentinel = backup / "resume_provenance.json"
            sentinel.write_bytes(b"preserve previous recovery")
            with patch.object(resume, "ROOT", root), patch.object(resume, "require_supported_ultralytics"), \
                    patch.object(resume, "require_no_live_training"):
                with self.assertRaises(FileExistsError):
                    resume.prepare(base / "source", base / "mixed.yaml", backup)
            self.assertEqual(sentinel.read_bytes(), b"preserve previous recovery")
            self.assertFalse((root / "runs").exists())

    def test_another_training_process_is_rejected_without_termination(self):
        other = SimpleNamespace(info={"pid": os.getpid() + 10000, "name": "python",
                                      "cmdline": ["python", "scripts/finetune_video.py", "train"]})
        fake_psutil = SimpleNamespace(process_iter=lambda fields: [other])
        with patch.dict("sys.modules", {"psutil": fake_psutil}):
            with self.assertRaisesRegex(RuntimeError, "Another training process"):
                resume.require_no_live_training()


if __name__ == "__main__":
    unittest.main()
