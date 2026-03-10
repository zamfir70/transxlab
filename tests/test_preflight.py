"""Tests for Level 1: Preflight checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from transxlab.knowledge.rules import Severity, TrainingContext, get_rules_for_context
from transxlab.knowledge.failures import get_preflight_failure_modes
from transxlab.knowledge.lessons import find_lesson_for_rule
from transxlab.preflight.config import ConfigReport, check_config, _detect_context
from transxlab.preflight.data import DataReport, check_data
from transxlab.preflight.environment import EnvironmentReport, check_environment
from transxlab.preflight.paths import PathReport, check_paths
from transxlab.preflight.resources import ResourceReport, estimate_resources, _parse_param_count


# ---------------------------------------------------------------------------
# Knowledge base tests
# ---------------------------------------------------------------------------

class TestKnowledgeRules:
    def test_rules_exist_for_all_contexts(self):
        for ctx in [TrainingContext.FINE_TUNE, TrainingContext.LORA, TrainingContext.SCRATCH]:
            rules = get_rules_for_context(ctx)
            assert len(rules) > 0, f"No rules for {ctx}"

    def test_lr_rule_fine_tune_catches_high_lr(self):
        rules = get_rules_for_context(TrainingContext.FINE_TUNE)
        lr_rules = [r for r in rules if r.parameter == "lr"]
        assert len(lr_rules) == 1
        result = lr_rules[0].check(3e-5)
        # 3e-5 is within range, should not trigger
        assert result is None
        # But 1e-4 should trigger (AC-v2 lesson: too high for fine-tuning)
        result = lr_rules[0].check(1e-4)
        assert result is not None
        assert result[0] == Severity.WARN

    def test_lr_rule_scratch_allows_higher(self):
        rules = get_rules_for_context(TrainingContext.SCRATCH)
        lr_rules = [r for r in rules if r.parameter == "lr"]
        assert len(lr_rules) == 1
        # 1e-3 should be fine for scratch
        result = lr_rules[0].check(1e-3)
        assert result is None

    def test_any_context_rules_included(self):
        """TrainingContext.ANY rules should appear in all contexts."""
        for ctx in [TrainingContext.FINE_TUNE, TrainingContext.LORA, TrainingContext.SCRATCH]:
            rules = get_rules_for_context(ctx)
            weight_decay_rules = [r for r in rules if r.parameter == "weight_decay"]
            assert len(weight_decay_rules) >= 1


class TestFailureModes:
    def test_preflight_failure_modes_exist(self):
        modes = get_preflight_failure_modes()
        assert len(modes) >= 5

    def test_ac_v2_failures_present(self):
        modes = get_preflight_failure_modes()
        names = [m.name for m in modes]
        assert "LR Too High (Fine-tune)" in names
        assert "Template Memorization" in names
        assert "Missing Diversity Signal" in names


class TestLessons:
    def test_ac_v2_lesson_exists(self):
        lesson = find_lesson_for_rule("LR Too High (Fine-tune)")
        assert lesson is not None
        assert lesson.id == "AC-v2"
        assert "$665" in lesson.cost

    def test_no_lesson_for_unknown_rule(self):
        assert find_lesson_for_rule("Nonexistent Rule") is None


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_python_version_detected(self):
        report = check_environment()
        python_checks = [c for c in report.checks if "Python" in c.name]
        assert len(python_checks) == 1
        assert python_checks[0].status in (Severity.INFO, Severity.WARN)

    def test_torch_check(self):
        report = check_environment()
        torch_checks = [c for c in report.checks if "PyTorch" in c.name]
        assert len(torch_checks) >= 1

    def test_required_packages(self):
        report = check_environment(required_packages=["json", "os"])
        pkg_checks = [c for c in report.checks if "Package" in c.name]
        assert len(pkg_checks) == 2
        assert all(c.status == Severity.INFO for c in pkg_checks)

    def test_missing_package_detected(self):
        report = check_environment(required_packages=["nonexistent_fake_package_xyz"])
        pkg_checks = [c for c in report.checks if "Package" in c.name]
        assert len(pkg_checks) == 1
        assert pkg_checks[0].status == Severity.FAIL


# ---------------------------------------------------------------------------
# Data validation tests
# ---------------------------------------------------------------------------

class TestDataValidation:
    def _make_jsonl(self, tmp: Path, name: str, records: list[dict]) -> Path:
        path = tmp / name
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_valid_data(self, tmp_path):
        train = self._make_jsonl(tmp_path, "train.jsonl", [
            {"input": "hello", "output": "world"},
            {"input": "foo", "output": "bar"},
        ])
        report = check_data(train_path=train)
        assert report.train_count == 2
        assert report.worst_severity == Severity.INFO or report.worst_severity == Severity.WARN
        # Should warn about missing val split
        val_checks = [c for c in report.checks if "Validation" in c.name or "val" in c.name.lower()]
        assert any(c.status == Severity.WARN for c in val_checks)

    def test_missing_field(self, tmp_path):
        train = self._make_jsonl(tmp_path, "train.jsonl", [
            {"question": "hello", "answer": "world"},
        ])
        report = check_data(train_path=train, input_field="input", output_field="output")
        assert report.worst_severity == Severity.FAIL

    def test_empty_file(self, tmp_path):
        train = tmp_path / "train.jsonl"
        train.write_text("")
        report = check_data(train_path=train)
        assert report.worst_severity == Severity.FAIL

    def test_overlap_detection(self, tmp_path):
        train = self._make_jsonl(tmp_path, "train.jsonl", [
            {"input": "hello", "output": "world"},
            {"input": "foo", "output": "bar"},
        ])
        val = self._make_jsonl(tmp_path, "val.jsonl", [
            {"input": "hello", "output": "world"},  # overlap!
        ])
        report = check_data(train_path=train, val_path=val)
        overlap_checks = [c for c in report.checks if "overlap" in c.name.lower()]
        assert len(overlap_checks) == 1
        assert overlap_checks[0].status == Severity.WARN

    def test_no_overlap(self, tmp_path):
        train = self._make_jsonl(tmp_path, "train.jsonl", [
            {"input": "hello", "output": "world"},
        ])
        val = self._make_jsonl(tmp_path, "val.jsonl", [
            {"input": "different", "output": "data"},
        ])
        report = check_data(train_path=train, val_path=val)
        overlap_checks = [c for c in report.checks if "overlap" in c.name.lower()]
        assert len(overlap_checks) == 1
        assert overlap_checks[0].status == Severity.INFO

    def test_auto_detect_splits(self, tmp_path):
        self._make_jsonl(tmp_path, "train.jsonl", [
            {"input": "a", "output": "b"},
        ])
        self._make_jsonl(tmp_path, "val.jsonl", [
            {"input": "c", "output": "d"},
        ])
        report = check_data(data_dir=tmp_path)
        assert report.train_count == 1
        assert report.val_count == 1

    def test_empty_examples_warned(self, tmp_path):
        train = self._make_jsonl(tmp_path, "train.jsonl", [
            {"input": "", "output": "world"},
            {"input": "foo", "output": ""},
        ])
        report = check_data(train_path=train)
        empty_checks = [c for c in report.checks if "Empty" in c.name or "empty" in c.name]
        assert any(c.status == Severity.WARN for c in empty_checks)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_detect_fine_tune(self):
        assert _detect_context({"method": "full"}) == TrainingContext.FINE_TUNE
        assert _detect_context({}) == TrainingContext.FINE_TUNE

    def test_detect_lora(self):
        assert _detect_context({"method": "lora"}) == TrainingContext.LORA
        assert _detect_context({"lora_r": 16}) == TrainingContext.LORA

    def test_detect_qlora(self):
        assert _detect_context({"method": "qlora"}) == TrainingContext.QLORA
        assert _detect_context({"lora_r": 16, "load_in_4bit": True}) == TrainingContext.QLORA

    def test_detect_scratch(self):
        assert _detect_context({"from_scratch": "true"}) == TrainingContext.SCRATCH

    def test_sane_config_passes(self):
        config = {"lr": 3e-5, "epochs": 3, "weight_decay": 0.01}
        report = check_config(config)
        assert report.worst_severity in (Severity.INFO, Severity.WARN)
        # lr should pass
        lr_checks = [c for c in report.checks if "lr" in c.name.lower()]
        assert any(c.status == Severity.INFO for c in lr_checks)

    def test_high_lr_caught(self):
        config = {"lr": 2e-4, "method": "full"}
        report = check_config(config)
        lr_checks = [c for c in report.checks if "lr" in c.name.lower()]
        assert any(c.status == Severity.WARN for c in lr_checks)

    def test_high_lr_ok_for_scratch(self):
        config = {"lr": 1e-3, "from_scratch": "true"}
        report = check_config(config)
        lr_checks = [c for c in report.checks if "lr" in c.name.lower()]
        assert all(c.status == Severity.INFO for c in lr_checks)

    def test_warmup_ratio_caught(self):
        config = {"warmup_steps": 500, "batch_size": 8, "epochs": 1}
        # 100 examples / batch 8 = 13 steps. 500 warmup >> 13 steps.
        report = check_config(config, n_examples=100)
        warmup_checks = [c for c in report.checks if "Warmup" in c.name or "warmup" in c.name]
        assert any(c.status == Severity.WARN for c in warmup_checks)

    def test_lora_alpha_ratio(self):
        config = {"method": "lora", "lora_r": 16, "lora_alpha": 128}
        report = check_config(config)
        ratio_checks = [c for c in report.checks if "alpha" in c.name.lower()]
        assert any(c.status == Severity.WARN for c in ratio_checks)

    def test_diversity_signal_warning(self):
        config = {
            "method": "lora",
            "lora_r": 16,
            "task_type": "creative generation",
            "diversity_loss_weight": 0.0,
        }
        report = check_config(config)
        div_checks = [c for c in report.checks if "diversity" in c.name.lower() or "Diversity" in c.name]
        assert any(c.status == Severity.WARN for c in div_checks)


# ---------------------------------------------------------------------------
# Path validation tests
# ---------------------------------------------------------------------------

class TestPathValidation:
    def test_existing_writable_dir(self, tmp_path):
        report = check_paths(output_dir=tmp_path)
        assert report.worst_severity == Severity.INFO

    def test_nonexistent_dir_warns(self, tmp_path):
        new_dir = tmp_path / "new_subdir"
        report = check_paths(output_dir=new_dir)
        dir_checks = [c for c in report.checks if "Output" in c.name]
        assert any(c.status == Severity.WARN for c in dir_checks)

    def test_fix_creates_dir(self, tmp_path):
        new_dir = tmp_path / "fixed_dir"
        report = check_paths(output_dir=new_dir, fix=True)
        assert new_dir.exists()
        dir_checks = [c for c in report.checks if "Output" in c.name]
        assert all(c.status == Severity.INFO for c in dir_checks)


# ---------------------------------------------------------------------------
# Resource estimation tests
# ---------------------------------------------------------------------------

class TestResources:
    def test_parse_param_count(self):
        assert _parse_param_count("7B") == 7_000_000_000
        assert _parse_param_count("125M") == 125_000_000
        assert _parse_param_count("3.8B") == 3_800_000_000

    def test_basic_estimation(self):
        config = {
            "total_params": "125M",
            "precision": "bf16",
            "batch_size": 4,
            "max_seq_len": 512,
            "epochs": 3,
        }
        report = estimate_resources(config, n_examples=1000)
        assert report.estimates.total_params == 125_000_000
        assert report.estimates.total_vram_gb > 0
        assert report.estimates.total_steps > 0

    def test_lora_reduces_optimizer_memory(self):
        full_config = {"total_params": "7B", "method": "full", "precision": "bf16"}
        lora_config = {"total_params": "7B", "method": "lora", "lora_r": 16, "precision": "bf16"}
        full_report = estimate_resources(full_config)
        lora_report = estimate_resources(lora_config)
        # LoRA should have much less optimizer memory
        assert lora_report.estimates.optimizer_memory_gb < full_report.estimates.optimizer_memory_gb

    def test_model_name_param_detection(self):
        config = {"model": "mistral-7b-instruct-v0.3", "precision": "bf16"}
        report = estimate_resources(config)
        assert report.estimates.total_params == 7_000_000_000

    def test_steps_calculated(self):
        config = {
            "total_params": "125M",
            "batch_size": 8,
            "grad_accum_steps": 4,
            "epochs": 3,
        }
        report = estimate_resources(config, n_examples=1000)
        # effective batch = 32, steps/epoch = ceil(1000/32) = 32
        assert report.estimates.steps_per_epoch == 32
        assert report.estimates.total_steps == 96


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReport:
    def test_markdown_report_saved(self, tmp_path):
        from transxlab.preflight.report import save_markdown_report
        env = check_environment()
        save_markdown_report(tmp_path / "report.md", env, None, None, None, None)
        assert (tmp_path / "report.md").exists()
        content = (tmp_path / "report.md").read_text()
        assert "TransXLab Preflight Report" in content

    def test_json_report_saved(self, tmp_path):
        from transxlab.preflight.report import save_json_report
        env = check_environment()
        save_json_report(tmp_path / "report.json", env, None, None, None, None)
        assert (tmp_path / "report.json").exists()
        data = json.loads((tmp_path / "report.json").read_text())
        assert "verdict" in data
        assert "environment" in data
