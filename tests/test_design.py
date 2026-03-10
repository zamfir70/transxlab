"""Tests for Level 2: Design interview, architecture, heuristics, spec."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from transxlab.design.architecture import ArchitectureSpec, recommend
from transxlab.design.heuristics import (
    recommend_lora_config,
    recommend_training_schedule,
    select_scratch_dimensions,
)
from transxlab.design.interview import (
    CreativityPriority,
    DesignInputs,
    InputFormat,
    OutputFormat,
    TaskType,
    TrainingApproach,
    TrainingMethod,
    load_interview_from_file,
    save_interview,
)
from transxlab.design.spec import build_spec_dict, save_design_report, save_spec


# ---------------------------------------------------------------------------
# Interview tests
# ---------------------------------------------------------------------------

class TestInterview:
    def test_design_inputs_serialization(self, tmp_path):
        inputs = DesignInputs(
            task_description="Generate hypotheses",
            task_type=TaskType.GENERATE,
            approach=TrainingApproach.FINE_TUNE,
            base_model="mistral-7b",
            training_method=TrainingMethod.LORA,
            vram_gb=32.0,
            data_size=5000,
            creativity=CreativityPriority.CREATIVITY,
        )
        path = tmp_path / "inputs.yaml"
        save_interview(inputs, path)
        loaded = load_interview_from_file(path)
        assert loaded.task_description == "Generate hypotheses"
        assert loaded.task_type == TaskType.GENERATE
        assert loaded.approach == TrainingApproach.FINE_TUNE
        assert loaded.base_model == "mistral-7b"
        assert loaded.training_method == TrainingMethod.LORA
        assert loaded.vram_gb == 32.0

    def test_to_dict_and_from_dict(self):
        inputs = DesignInputs(task_type=TaskType.CLASSIFY, approach=TrainingApproach.SCRATCH)
        d = inputs.to_dict()
        assert d["task_type"] == "classify"
        assert d["approach"] == "scratch"
        restored = DesignInputs.from_dict(d)
        assert restored.task_type == TaskType.CLASSIFY
        assert restored.approach == TrainingApproach.SCRATCH


# ---------------------------------------------------------------------------
# Heuristics tests
# ---------------------------------------------------------------------------

class TestHeuristics:
    def test_scratch_dimensions_tiny(self):
        dims = select_scratch_dimensions("25M")
        assert dims.tier == "tiny"
        assert dims.d_model == 256

    def test_scratch_dimensions_small(self):
        dims = select_scratch_dimensions("125M")
        assert dims.tier == "small"
        assert dims.d_model == 512

    def test_scratch_dimensions_medium(self):
        dims = select_scratch_dimensions("350M")
        assert dims.tier == "medium"
        assert dims.d_model == 768

    def test_scratch_dimensions_large(self):
        dims = select_scratch_dimensions("770M")
        assert dims.tier == "large"
        assert dims.d_model == 1024

    def test_training_schedule_finetune(self):
        schedule = recommend_training_schedule(
            approach="fine-tune",
            method="lora",
            data_size=5000,
            param_count=7_000_000_000,
            vram_gb=32.0,
        )
        assert schedule.lr == 2e-4  # LoRA lr
        assert schedule.epochs >= 1
        assert schedule.precision == "bf16"
        assert schedule.effective_batch == schedule.batch_size * schedule.grad_accum_steps

    def test_training_schedule_scratch(self):
        schedule = recommend_training_schedule(
            approach="scratch",
            method="full",
            data_size=50000,
            param_count=125_000_000,
            vram_gb=32.0,
        )
        assert schedule.lr >= 1e-4  # Higher for scratch
        assert schedule.epochs >= 3
        assert schedule.warmup_steps >= 100

    def test_lora_config_creative(self):
        config = recommend_lora_config("generate", "creativity", "mistral-7b")
        assert config.r == 32  # Higher rank for creative
        assert config.alpha == 64  # 2 * r
        assert len(config.target_modules) > 0

    def test_lora_config_classify(self):
        config = recommend_lora_config("classify", "consistency", "flan-t5-xl")
        assert config.r == 8  # Lower for classification
        assert "q" in config.target_modules  # T5 style


# ---------------------------------------------------------------------------
# Architecture recommendation tests
# ---------------------------------------------------------------------------

class TestArchitecture:
    def _make_finetune_inputs(self, **overrides) -> DesignInputs:
        defaults = dict(
            task_description="Generate creative hypotheses",
            task_type=TaskType.GENERATE,
            input_format=InputFormat.STRUCTURED,
            output_format=OutputFormat.TEXT,
            approach=TrainingApproach.FINE_TUNE,
            base_model="recommend",
            training_method=TrainingMethod.LORA,
            vram_gb=32.0,
            data_size=5000,
            creativity=CreativityPriority.CREATIVITY,
        )
        defaults.update(overrides)
        return DesignInputs(**defaults)

    def _make_scratch_inputs(self, **overrides) -> DesignInputs:
        defaults = dict(
            task_description="Structured generation",
            task_type=TaskType.GENERATE,
            input_format=InputFormat.STRUCTURED,
            output_format=OutputFormat.TEXT,
            approach=TrainingApproach.SCRATCH,
            param_budget="125M",
            input_seq_len=512,
            output_seq_len=256,
            vram_gb=32.0,
            data_size=50000,
        )
        defaults.update(overrides)
        return DesignInputs(**defaults)

    def test_finetune_generates_recommendations(self):
        inputs = self._make_finetune_inputs()
        spec = recommend(inputs)
        assert len(spec.recommendations) > 0
        assert spec.base_model != ""
        assert spec.training_method == "lora"
        assert spec.lora_config is not None
        assert spec.schedule is not None

    def test_finetune_diversity_loss_for_creative(self):
        inputs = self._make_finetune_inputs(creativity=CreativityPriority.CREATIVITY)
        spec = recommend(inputs)
        assert any("diversity" in str(r.value).lower() for r in spec.recommendations)

    def test_finetune_no_diversity_for_classify(self):
        inputs = self._make_finetune_inputs(
            task_type=TaskType.CLASSIFY,
            output_format=OutputFormat.CLASSES,
            creativity=CreativityPriority.CONSISTENCY,
        )
        spec = recommend(inputs)
        loss_recs = [r for r in spec.recommendations if r.key == "loss"]
        assert not any("diversity" in str(r.value).lower() for r in loss_recs)

    def test_scratch_generates_recommendations(self):
        inputs = self._make_scratch_inputs()
        spec = recommend(inputs)
        assert len(spec.recommendations) > 0
        assert spec.scratch_dims is not None
        assert spec.schedule is not None
        assert spec.architecture_type in ("encoder-decoder", "decoder-only")

    def test_scratch_warns_insufficient_data(self):
        inputs = self._make_scratch_inputs(data_size=500)
        spec = recommend(inputs)
        data_recs = [r for r in spec.recommendations if "data" in r.key.lower()]
        assert len(data_recs) > 0  # Should warn about insufficient data

    def test_scratch_encoder_only_for_classify(self):
        inputs = self._make_scratch_inputs(
            task_type=TaskType.CLASSIFY,
            output_format=OutputFormat.CLASSES,
        )
        spec = recommend(inputs)
        assert spec.architecture_type == "encoder-only"

    def test_vram_estimate_present(self):
        inputs = self._make_finetune_inputs()
        spec = recommend(inputs)
        assert spec.estimated_vram_gb > 0


# ---------------------------------------------------------------------------
# Spec output tests
# ---------------------------------------------------------------------------

class TestSpec:
    def test_build_spec_dict(self):
        inputs = DesignInputs(
            task_description="Test task",
            task_type=TaskType.GENERATE,
            approach=TrainingApproach.FINE_TUNE,
            base_model="mistral-7b",
            training_method=TrainingMethod.LORA,
            vram_gb=32.0,
            data_size=5000,
        )
        spec = recommend(inputs)
        d = build_spec_dict(inputs, spec)

        assert "task" in d
        assert "architecture" in d
        assert "training" in d
        assert "estimates" in d
        assert "transxform" in d
        assert d["task"]["type"] == "generate"

    def test_save_spec_yaml(self, tmp_path):
        inputs = DesignInputs(
            task_type=TaskType.GENERATE,
            approach=TrainingApproach.FINE_TUNE,
            base_model="mistral-7b",
            training_method=TrainingMethod.LORA,
        )
        spec = recommend(inputs)
        d = build_spec_dict(inputs, spec)
        path = tmp_path / "spec.yaml"
        save_spec(d, path)
        assert path.exists()
        loaded = yaml.safe_load(path.read_text())
        assert "task" in loaded

    def test_save_design_report(self, tmp_path):
        inputs = DesignInputs(
            task_description="Test",
            task_type=TaskType.GENERATE,
            approach=TrainingApproach.FINE_TUNE,
        )
        spec = recommend(inputs)
        path = tmp_path / "report.md"
        save_design_report(path, inputs, spec)
        assert path.exists()
        content = path.read_text()
        assert "Design Report" in content

    def test_transxform_spec_for_generation(self):
        inputs = DesignInputs(
            task_type=TaskType.GENERATE,
            approach=TrainingApproach.FINE_TUNE,
        )
        spec = recommend(inputs)
        d = build_spec_dict(inputs, spec)
        tf = d["transxform"]
        assert "invariants" in tf
        assert "eval" in tf
        assert "generation_quality" in tf["eval"]["metrics"]

    def test_transxform_spec_for_classification(self):
        inputs = DesignInputs(
            task_type=TaskType.CLASSIFY,
            output_format=OutputFormat.CLASSES,
            approach=TrainingApproach.FINE_TUNE,
        )
        spec = recommend(inputs)
        d = build_spec_dict(inputs, spec)
        tf = d["transxform"]
        assert "accuracy" in tf["eval"]["metrics"]
