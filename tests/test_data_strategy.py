"""Tests for Level 3: Data quality analysis, strategy, and sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transxlab.data_strategy.analyzer import QualityReport, analyze_quality
from transxlab.data_strategy.sources import suggest_sources
from transxlab.data_strategy.strategy import (
    DataStrategy,
    get_all_strategies,
    recommend_strategy,
)


# ---------------------------------------------------------------------------
# Quality analyzer tests
# ---------------------------------------------------------------------------

class TestQualityAnalyzer:
    def _make_examples(self, outputs: list[str]) -> list[dict]:
        return [{"input": f"input_{i}", "output": o} for i, o in enumerate(outputs)]

    def test_diverse_outputs(self):
        examples = self._make_examples([
            "The cat sat on the mat and looked out the window.",
            "Machine learning transforms data into predictions.",
            "The ocean waves crashed against the rocky shore.",
            "A good algorithm balances speed and accuracy.",
            "The forest was quiet except for birdsong.",
            "Neural networks learn hierarchical representations.",
            "The city lights reflected off the wet pavement.",
            "Data preprocessing is crucial for model performance.",
            "The mountain trail wound through ancient trees.",
            "Transfer learning leverages pretrained knowledge.",
        ])
        report = analyze_quality(examples, output_field="output", task_type="creative")
        assert report.metrics.self_bleu is not None
        assert report.metrics.self_bleu < 0.5  # Diverse outputs should have low self-BLEU

    def test_templated_outputs_detected(self):
        # Create highly templated outputs
        examples = self._make_examples([
            f"The entity {i} is related to entity {i+1} through relationship type A."
            for i in range(50)
        ])
        report = analyze_quality(examples, output_field="output", task_type="creative")
        assert report.metrics.self_bleu is not None
        assert report.metrics.self_bleu > 0.3  # Templated should have higher self-BLEU

    def test_duplication_detected(self):
        examples = self._make_examples([
            "Exactly the same output.",
            "Exactly the same output.",
            "Exactly the same output.",
            "Exactly the same output.",
            "Exactly the same output.",
            "Something different here.",
            "Another unique output.",
            "Yet another one.",
            "One more unique text.",
            "And the last one.",
        ])
        report = analyze_quality(examples, output_field="output")
        assert report.metrics.duplication_rate > 0.3  # 4 duplicates out of 10

    def test_empty_outputs_flagged(self):
        examples = self._make_examples(["", "", "", "something", "another"])
        report = analyze_quality(examples, output_field="output")
        assert report.metrics.empty_rate > 0.5
        assert any(c.name == "Empty outputs" for c in report.checks)

    def test_length_stats(self):
        examples = self._make_examples([
            "Short.",
            "A medium length sentence with several words.",
            "A much longer sentence that contains many more words and goes on for quite a while.",
        ])
        report = analyze_quality(examples, output_field="output")
        assert report.metrics.length_mean > 0
        assert report.metrics.length_std > 0

    def test_empty_examples_list(self):
        report = analyze_quality([], output_field="output")
        assert report.worst_severity.value in ("warn", "fail")

    def test_large_sample_cap(self):
        examples = self._make_examples([f"Output number {i}" for i in range(10000)])
        report = analyze_quality(examples, output_field="output", sample_size=500)
        assert report.metrics.n_analyzed == 500


# ---------------------------------------------------------------------------
# Strategy recommender tests
# ---------------------------------------------------------------------------

class TestStrategy:
    def test_creative_gets_distillation(self):
        strategy = recommend_strategy("creative generation")
        assert strategy.name == "Distillation"

    def test_classify_gets_labeled(self):
        strategy = recommend_strategy("classification")
        assert strategy.name == "Labeled Examples"

    def test_instruction_gets_self_instruct(self):
        strategy = recommend_strategy("instruction following")
        assert strategy.name == "Self-Instruct"

    def test_domain_gets_corpus(self):
        strategy = recommend_strategy("domain adaptation")
        assert strategy.name == "Domain Corpus"

    def test_structured_gets_synthetic(self):
        strategy = recommend_strategy("structured output")
        assert strategy.name == "Synthetic Structured"

    def test_default_is_distillation(self):
        strategy = recommend_strategy("unknown task")
        assert strategy.name == "Distillation"

    def test_all_strategies_have_required_fields(self):
        for key, strat in get_all_strategies().items():
            assert strat.name, f"{key} missing name"
            assert strat.description, f"{key} missing description"
            assert strat.minimum_examples > 0, f"{key} missing minimum_examples"
            assert strat.quality_checks, f"{key} missing quality_checks"
            assert strat.anti_patterns, f"{key} missing anti_patterns"


# ---------------------------------------------------------------------------
# Source suggestion tests
# ---------------------------------------------------------------------------

class TestSources:
    def test_relational_task_suggests_conceptnet(self):
        sources = suggest_sources(task_type="relational reasoning", domain="relational")
        names = [s.name for s in sources]
        assert "ConceptNet" in names

    def test_code_task_suggests_stack(self):
        sources = suggest_sources(task_type="code generation", domain="code")
        names = [s.name for s in sources]
        assert "The Stack" in names

    def test_always_includes_distillation(self):
        sources = suggest_sources(task_type="anything")
        names = [s.name for s in sources]
        assert "Custom Distillation" in names

    def test_nli_has_warnings(self):
        sources = suggest_sources(task_type="classification", domain="classification")
        nli_sources = [s for s in sources if "NLI" in s.name]
        if nli_sources:
            assert len(nli_sources[0].warnings) > 0  # Should warn about template risk

    def test_empty_query_returns_results(self):
        sources = suggest_sources()
        assert len(sources) > 0  # At least custom distillation
