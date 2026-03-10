# TransXLab Implementation Plan

> TransXLab: The training architect. "Don't start what you can't finish."

This document is the working plan for implementing TransXLab across three levels.
TransXform (Rust, `d:\TransXform`) handles supervision *during* training.
TransXLab handles everything *before* training — from architecture design to preflight checks.

---

## Language Decision

**Python.** TransXLab's value is in the ML ecosystem (PyTorch, tokenizers, CUDA introspection, dataset loading). Rust would fight every library call. Python with type hints and a CLI framework (Click or Typer) is the right choice.

---

## Project Structure

```
TransXLab/
├── transxlab/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (Typer)
│   ├── preflight/              # Level 1
│   │   ├── __init__.py
│   │   ├── environment.py      # CUDA, Python, torch, GPU checks
│   │   ├── data.py             # Dataset validation
│   │   ├── config.py           # Hyperparameter sanity checks
│   │   ├── paths.py            # Output dir, checkpoint path validation
│   │   ├── resources.py        # VRAM estimation, time/cost estimation
│   │   └── report.py           # Preflight report generation
│   ├── design/                 # Level 2
│   │   ├── __init__.py
│   │   ├── interview.py        # Interactive design interview
│   │   ├── architecture.py     # Architecture recommendation engine
│   │   ├── heuristics.py       # Hyperparameter heuristic rules
│   │   └── spec.py             # Design spec generation (YAML output)
│   ├── data_strategy/          # Level 3
│   │   ├── __init__.py
│   │   ├── analyzer.py         # Data quality analysis (self-BLEU, diversity)
│   │   ├── strategy.py         # Data strategy recommender
│   │   └── sources.py          # Data source suggestions
│   └── knowledge/              # Shared knowledge base
│       ├── __init__.py
│       ├── models.py           # Known model specs (param counts, VRAM, etc.)
│       ├── rules.py            # Heuristic rules (lr ranges, batch sizing, etc.)
│       ├── failures.py         # Known failure modes and their signatures
│       └── lessons.py          # Hard-won lessons from real training failures
├── tests/
│   ├── test_preflight.py
│   ├── test_design.py
│   └── test_data_strategy.py
├── pyproject.toml
├── PLAN.md                     # This file
└── TransXLab Concept.md        # Original concept
```

---

## Level 1: Preflight

**Goal:** Validate that a training run is sane before it starts.
**Entry point:** `transxlab --setup [--config config.yaml] [--data-dir ./data]`
**Output:** `preflight_report.md` + pass/fail exit code

### 1.1 Environment Checks (`preflight/environment.py`)

| Check | How | Severity |
|-------|-----|----------|
| Python version | `sys.version_info` | warn if < 3.10 |
| PyTorch installed | `import torch` | fail |
| PyTorch version | `torch.__version__` | info |
| CUDA available | `torch.cuda.is_available()` | warn (CPU training is slow, not broken) |
| CUDA version | `torch.version.cuda` | info |
| GPU name + VRAM | `torch.cuda.get_device_properties(0)` | info |
| cuDNN available | `torch.backends.cudnn.is_available()` | warn |
| bf16 support | `torch.cuda.is_bf16_supported()` | info (affects precision recommendation) |
| Key packages | Import check: transformers, datasets, peft, etc. | fail if required by config |

**Implementation notes:**
- Return a structured `EnvironmentReport` dataclass, not just prints.
- Each check returns `(status: pass|warn|fail, message: str, detail: str)`.
- Fail fast on hard failures (no torch), continue collecting on warns/info.

### 1.2 Data Validation (`preflight/data.py`)

| Check | How | Severity |
|-------|-----|----------|
| Files exist | `Path.exists()` | fail |
| Files parse | Try load as JSON/JSONL/CSV/Parquet | fail |
| Required fields present | Check against config's `input_field`, `output_field` | fail |
| Train/val/test split exists | Check for separate files or split config | warn |
| No train/val overlap | Hash-based dedup check on input fields | warn |
| Example count | Count per split | info |
| Empty examples | Check for blank input/output | warn |
| Sequence length distribution | Tokenize sample, report p50/p95/max | info |
| Format consistency | All examples have same schema | warn |

**Implementation notes:**
- Support formats: JSONL (primary), JSON, CSV, Parquet, HuggingFace datasets.
- For large datasets (>100K), sample 10K for expensive checks (tokenization, overlap).
- Overlap check: hash the input field of each example, check set intersection.
- Sequence length check requires a tokenizer — use the one specified in config, or `AutoTokenizer` from the model name.

### 1.3 Config Validation (`preflight/config.py`)

Accepts a YAML/JSON config file. Validates hyperparameters against known-sane ranges.

| Parameter | Sane Range | Context |
|-----------|-----------|---------|
| `lr` (fine-tuning) | 1e-6 to 1e-4 | Higher risks catastrophic forgetting |
| `lr` (LoRA) | 1e-5 to 5e-4 | Adapters can tolerate slightly higher |
| `lr` (scratch) | 1e-4 to 1e-3 | Needs to be higher for random init |
| `batch_size` | Must fit VRAM (see 1.5) | Cross-ref with resource estimation |
| `epochs` (fine-tuning) | 1-5 | More risks overfitting on small data |
| `epochs` (scratch) | 3-50 | Depends on dataset size |
| `warmup_steps` | 0.01-0.1 of total steps | Too much warmup wastes steps |
| `weight_decay` | 0.0 to 0.1 | Higher than 0.1 is unusual |
| `grad_accum_steps` | 1-64 | effective_batch = batch × accum |
| `max_seq_len` | Must be ≤ model's max | Fail if exceeds model limit |
| `lora_r` | 4-64 | Higher = more params, more capacity |
| `lora_alpha` | Typically 2× `lora_r` | Common convention |

**Implementation notes:**
- Rules are defined declaratively in `knowledge/rules.py` so they're easy to update.
- Each rule has: `parameter`, `range`, `context` (fine-tune/scratch/lora), `severity`, `message`.
- Config can be a flat YAML or a nested one (support both).
- If no config provided, TransXLab can generate a default config from the design spec (Level 2 integration).

### 1.4 Path Validation (`preflight/paths.py`)

| Check | How | Severity |
|-------|-----|----------|
| Output dir exists or is creatable | `Path.mkdir(parents=True, exist_ok=True)` | fail |
| Output dir is writable | Write and delete a temp file | fail |
| Checkpoint naming won't collide | Check for existing files matching pattern | warn |
| Disk space sufficient | `shutil.disk_usage()` vs estimated checkpoint size | warn |

### 1.5 Resource Estimation (`preflight/resources.py`)

| Estimate | How |
|----------|-----|
| Parameters | From model config or `model.num_parameters()` |
| VRAM (training) | `params × bytes_per_param × overhead_factor` — bf16 = 2 bytes, optimizer states ~3× for AdamW, activations depend on batch×seq_len |
| VRAM (inference) | `params × bytes_per_param` + KV cache |
| Steps/epoch | `ceil(n_examples / effective_batch)` |
| Total steps | `steps_per_epoch × epochs` |
| Time estimate | `total_steps × ms_per_step` (benchmark 10 steps or use heuristic) |
| Cost estimate | `$0` for local, or `time × $/hr` for cloud GPU |

**VRAM estimation formula (training, AdamW, bf16):**
```
model_memory = params × 2  (bf16 weights)
optimizer_memory = params × 8  (AdamW: 2 fp32 copies + 2 momentum states)
gradient_memory = params × 2  (bf16 gradients)
activation_memory ≈ batch × seq_len × hidden × n_layers × 2  (rough)
total ≈ model + optimizer + gradient + activation
```

**VRAM estimation formula (training, AdamW, bf16, LoRA):**
```
base_model_memory = total_params × 2         (bf16 weights, frozen)
adapter_memory = adapter_params × 2          (bf16 adapter weights)
optimizer_memory = adapter_params × 8        (AdamW states for adapters ONLY)
gradient_memory = adapter_params × 2         (bf16 gradients for adapters ONLY)
activation_memory ≈ batch × seq_len × hidden × n_layers × 2  (rough, same as full)
total ≈ base_model + adapter + optimizer + gradient + activation
```

Key difference: full fine-tuning pays optimizer/gradient cost on ALL params.
LoRA pays it only on adapter params (typically 0.1-1% of total). This is why
LoRA fits on much smaller GPUs despite loading the same base model.

**Implementation notes:**
- Build a `VRAMEstimator` that takes model config + training config → estimated GB.
- Separate code paths for full fine-tune vs LoRA vs QLoRA (4-bit base weights).
- Compare against actual GPU VRAM. Flag if estimated > 90% of available.
- Time estimation is rough — offer to run a 10-step benchmark for accuracy.

### 1.6 Preflight Report (`preflight/report.py`)

Generate a markdown report summarizing all checks. Format matches the example in the concept doc.

Sections:
1. **Environment** — Python, torch, CUDA, GPU
2. **Data** — counts, schema, overlap check
3. **Config** — parameter validation results
4. **Paths** — output dir, checkpoint naming
5. **Estimates** — VRAM, time, cost, steps
6. **Verdict** — READY / WARNINGS / BLOCKED (with reasons)
7. **Command** — The exact command to run training

Exit codes: `0` = ready, `1` = warnings (proceed with caution), `2` = blocked (fix issues first).

---

## Level 2: Design

**Goal:** Recommend architecture and training configuration based on task description.
**Entry point:** `transxlab --design`
**Output:** `design_spec.yaml` + `design_report.md`

### 2.1 Design Interview (`design/interview.py`)

Interactive CLI interview. Questions adapt based on previous answers.

**Core questions (always asked):**

| # | Question | Why |
|---|----------|-----|
| 1 | What's the task? (classify / generate / embed / other) | Determines architecture family |
| 2 | What's the input format? (text / structured / multimodal) | Determines encoder needs |
| 3 | What's the output format? (classes / text / embeddings) | Determines decoder/head |
| 4 | Fine-tune or scratch? | Determines the entire recommendation path |

**If fine-tuning:**

| # | Question | Why |
|---|----------|-----|
| 5 | Base model preference? (or "recommend") | Anchor for all recommendations |
| 6 | Training method? (full / LoRA / QLoRA) | Determines VRAM and param strategy |
| 7 | VRAM constraint? | Filters viable options |
| 8 | Training data size? | Affects epochs, lr schedule |
| 9 | Creativity vs. consistency priority? | Affects temperature, sampling, diversity loss |

**If scratch:**

| # | Question | Why |
|---|----------|-----|
| 5 | Parameter budget? | Determines model dimensions |
| 6 | Sequence length budget? (input, output) | Determines positional encoding, memory |
| 7 | VRAM constraint? | Cross-ref with param budget |
| 8 | Training data size? | Determines if scratch is even viable |
| 9 | Latency constraint? | Affects model size, quantization |

**Implementation notes:**
- Use `rich` or `questionary` for interactive prompts with defaults.
- Support a `--non-interactive` mode that reads answers from a YAML file (for automation).
- Store interview responses in a structured `DesignInputs` dataclass.
- Allow re-running with `--resume` to modify previous answers.

### 2.2 Architecture Recommendation (`design/architecture.py`)

Decision engine that maps interview answers to architecture specs.

**Decision tree (fine-tuning path):**

```
task_type + vram_constraint + data_size → base_model_recommendation
base_model + method → freeze_strategy, lora_config
base_model + task → recommended_lr, epochs, warmup
```

**Decision tree (scratch path):**

```
task_modality → architecture_family (enc-only, dec-only, enc-dec)
param_budget + architecture_family → dimensions (d_model, n_heads, n_layers, d_ff)
data_size → training_schedule (lr, epochs, warmup)
task_type → loss_function, output_head
```

**Known model database (`knowledge/models.py`):**

Store specs for commonly used models:

```python
MODELS = {
    "mistral-7b": ModelSpec(
        params="7B", vram_bf16="14GB", vram_qlora="6GB",
        max_seq_len=32768, architecture="decoder-only",
        good_for=["generation", "instruction-following"],
    ),
    "flan-t5-xl": ModelSpec(
        params="3B", vram_bf16="6GB", vram_qlora="3GB",
        max_seq_len=512, architecture="encoder-decoder",
        good_for=["classification", "structured-generation"],
    ),
    # ... more models
}
```

**Dimension heuristics for scratch (`knowledge/rules.py`):**

```python
SCRATCH_HEURISTICS = {
    # param_budget → (d_model, n_heads, n_layers_enc, n_layers_dec, d_ff)
    "tiny":   (256,  4,  4, 4, 1024),    # ~25M
    "small":  (512,  8,  6, 6, 2048),    # ~125M
    "medium": (768,  12, 12, 12, 3072),  # ~350M
    "large":  (1024, 16, 24, 24, 4096),  # ~770M
}
```

Select based on parameter budget, then adjust d_ff if needed to hit target.

**Implementation notes:**
- Recommendations include a **confidence level** and **rationale** for each choice.
- If the system is unsure, say so: "This is outside common patterns. Consider consulting literature on X."
- Always explain *why* — don't just emit numbers.

### 2.3 Design Spec Output (`design/spec.py`)

Emit `design_spec.yaml` (machine-readable) and `design_report.md` (human-readable).

**design_spec.yaml contents:**
- `task`: description, type, modality
- `architecture`: type, dimensions, config
- `training`: lr, epochs, warmup, batch, precision, method
- `data`: expected format, split ratios, minimum size
- `estimates`: params, VRAM, time
- `rationale`: why each choice was made (stored as comments or a separate section)

**design_report.md:** Narrative version of the spec with explanations. Similar to the "TransXlab recommends" block in the concept doc.

The design spec feeds directly into Level 1's `--setup` command — TransXLab can validate data and config *against the design spec*.

---

## Level 3: Data Strategy

**Goal:** Analyze existing data quality and recommend data sourcing strategies.
**Entry point:** `transxlab --data [--design-spec design_spec.yaml] [--data-dir ./data]`
**Output:** `data_plan.md` + quality metrics

### 3.1 Data Quality Analyzer (`data_strategy/analyzer.py`)

Goes beyond Level 1's "does the data parse" to "is the data *good*."

| Metric | What It Measures | How |
|--------|-----------------|-----|
| Self-BLEU | Template contamination | Pairwise BLEU on output samples — high = templated |
| Lexical diversity | Vocabulary richness | Type-token ratio on outputs |
| Length distribution | Output length variance | Std dev of token counts |
| Type diversity | Category balance | Count unique output types/classes |
| Duplication rate | Near-duplicate examples | MinHash or exact-match on inputs |
| Format compliance | Outputs match expected schema | Parse attempt on each output |

**Thresholds (from `knowledge/rules.py`):**

| Metric | Good | Concerning | Bad |
|--------|------|-----------|-----|
| Self-BLEU (creative) | < 0.3 | 0.3-0.6 | > 0.6 |
| Self-BLEU (structured) | < 0.5 | 0.5-0.7 | > 0.7 |
| Duplication rate | < 1% | 1-5% | > 5% |
| Lexical diversity (TTR) | > 0.3 | 0.2-0.3 | < 0.2 |

**Implementation notes:**
- Self-BLEU: Sample 5K outputs (1K is too small for 100K+ datasets), compute pairwise 4-gram BLEU, average.
- MinHash dedup: Use `datasketch` library or implement simple 128-hash MinHash.
- Run on a sample for large datasets (>50K examples).
- Report includes per-metric scores with pass/warn/fail status.

### 3.2 Data Strategy Recommender (`data_strategy/strategy.py`)

Given the task (from design spec or interview), recommend how to get and prepare data.

**Strategy selection logic:**

| Task Type | Primary Strategy | Fallback |
|-----------|-----------------|----------|
| Creative generation | Distillation from large model | Manual curation |
| Classification | Labeled examples (manual or crowd) | Zero-shot → label propagation |
| Instruction following | Distillation + human preference | Self-instruct |
| Domain adaptation | Domain corpus + continued pretraining | Domain-specific fine-tune data |
| Structured output | Schema-validated examples | Synthetic generation |

**For each strategy, recommend:**
- Minimum dataset size (with rationale)
- Format specification
- Quality checks to run
- Anti-patterns to avoid
- Suggested split ratios (train/val/test)

### 3.3 Data Source Suggestions (`data_strategy/sources.py`)

Suggest concrete data sources based on task domain.

| Domain | Potential Sources |
|--------|------------------|
| NLP general | HuggingFace datasets, Common Crawl subsets |
| Relational/knowledge | ConceptNet, WordNet, Wikidata |
| Code | The Stack, CodeSearchNet |
| Scientific | S2ORC, PubMed abstracts |
| Conversation | ShareGPT, LMSYS-Chat |
| Custom/domain | "Use distillation: prompt a large model with your task" |

**Implementation notes:**
- This is advisory, not automated downloading.
- Flag known-problematic datasets (NLI → template risk, etc.).
- Suggest data augmentation techniques where applicable.

---

## Knowledge Base (`knowledge/`)

The shared knowledge base underpins all three levels. It should be:
- **Declarative:** Rules as data structures, not hardcoded if/else.
- **Versioned:** Easy to update as we learn more.
- **Sourced:** Each rule should note where it came from (paper, empirical, lesson learned).

### `knowledge/rules.py`
```python
@dataclass
class HyperparamRule:
    parameter: str
    context: str          # "fine-tune", "lora", "scratch"
    min_val: float
    max_val: float
    recommended: float
    severity: str         # "fail", "warn", "info"
    rationale: str
    source: str           # "empirical", "literature", "AC-v2-postmortem"
```

### `knowledge/models.py`
Known model specs — params, VRAM, max_seq_len, architecture, strengths.

### `knowledge/failures.py`

Structured failure mode signatures. Each failure mode is a named pattern with detection rules,
consequences, and mitigations — so TransXLab can say *"Warning: This configuration matches
the signature for 'Template Memorization' (AC-v2-postmortem)."*

```python
@dataclass
class FailureMode:
    name: str
    detection_phase: str       # "preflight", "during_training", "post_training"
    signals: list[str]         # ["self_bleu > 0.6", "lr > 5e-5 for fine-tune"]
    consequence: str           # "Model memorizes templates instead of learning generation"
    mitigation: str            # "Reduce self-BLEU via distillation with diverse prompts"
    source: str                # "AC-v2-postmortem"
```

**Initial failure catalog:**

| Name | Phase | Signals | Consequence | Source |
|------|-------|---------|-------------|--------|
| LR Too High (Fine-tune) | preflight | `lr > 5e-5` for fine-tune context | Catastrophic forgetting, loss divergence | AC-v2 |
| Template Memorization | preflight | `self_bleu > 0.6` in training data | Model reproduces templates, no real generation | AC-v2 |
| Missing Diversity Signal | preflight | `diversity_loss_weight == 0` for creative task | Mode collapse, repetitive outputs | AC-v2 |
| Train/Val Overlap | preflight | Set intersection > 0 on input hashes | Inflated val metrics, false confidence | Literature |
| Warmup Too Long | preflight | `warmup_steps > 0.15 × total_steps` | Wasted compute, delayed learning | Empirical |
| VRAM Overflow | preflight | Estimated VRAM > 95% of available | OOM crash mid-training, lost progress | Empirical |
| Insufficient Data (Scratch) | preflight | `data_size < 10K` for scratch build | Underfitting, poor generalization | Literature |
| Wrong Eval Metric | preflight | Generation task with only loss-based eval | No signal on actual output quality | AC-v2 |

### `knowledge/lessons.py`

Hard-won lessons encoded from real training failures. Referenced in warnings.

```python
LESSONS = {
    "AC-v2": {
        "id": "AC-v2",
        "cost": "$665",
        "summary": "Abductive Completor v2 fine-tuning failures",
        "findings": {
            "lr": "1e-4 is too high for fine-tuning Flan-T5-XL. Use 1e-5 to 5e-5.",
            "diversity_loss": "For creative generation, diversity_loss_weight=0.0 causes mode collapse. Use >= 0.3.",
            "monitoring": "val_loss alone is insufficient. Must eval generation quality on novel queries.",
            "data": "Templated data (self-BLEU > 0.6) teaches templates, not generation. Use distillation.",
        },
        "rules_derived": [
            "LR Too High (Fine-tune)",
            "Template Memorization",
            "Missing Diversity Signal",
            "Wrong Eval Metric",
        ],
    },
}
```

Warnings reference lessons directly: *"Warning: lr=1e-4 for fine-tuning. See lesson AC-v2: '1e-4 is too high for fine-tuning Flan-T5-XL.'"*

---

## TransXform Handoff Spec

When TransXLab finishes its work, it emits a `transxform_spec` section in the config that TransXform
can consume directly. This closes the loop between architect and supervisor.

**What TransXLab emits for TransXform:**

```yaml
# transxform section in config.yaml
transxform:
  invariants:
    - "grad_norm < 10.0"
    - "loss < 15.0 after step 100"
  alerts:
    - "pw_cos > 0.9 for 3 consecutive evals → representation collapse"
  early_stop:
    metric: "novel_query_generation_quality"
    patience: 3
  checkpoints:
    save_every: 500
    keep_best: 3
  eval:
    novel_query_eval_every: 500    # For generation tasks
    metrics: ["loss", "generation_quality", "diversity"]
```

**How TransXLab decides what to include:**

| Task Type | Invariants | Alerts | Eval |
|-----------|-----------|--------|------|
| Fine-tune (generation) | grad_norm, loss floor | pw_cos collapse | Novel query generation |
| Fine-tune (classification) | grad_norm, loss floor | accuracy plateau | Val accuracy |
| Scratch (any) | grad_norm, loss floor, lr schedule | loss divergence | Task-specific |
| Creative generation | All above + diversity | mode collapse (self-BLEU on outputs) | Diversity + quality |

The TransXform spec is **derived from the design spec** — TransXLab knows the task type,
the failure modes to watch for, and the eval metrics that actually matter. This is the key
integration point: TransXLab tells TransXform *what to watch*, not just *what to run*.

---

## CLI Design

```
transxlab --design                    # Level 2: Interactive design interview
transxlab --data [--design-spec X]    # Level 3: Data analysis + strategy
transxlab --setup [--config X]        # Level 1: Preflight validation
transxlab --full                      # All three in sequence
```

Each command can run independently. When run together, outputs chain:
`--design` emits `design_spec.yaml` → `--data` validates against it → `--setup` generates `config.yaml` from it.

**Global flags:**
- `--output-dir` — where to write reports (default: `./transxlab_output/`)
- `--non-interactive` — read all inputs from files, no prompts
- `--verbose` / `--quiet` — control output detail
- `--json` — emit machine-readable output (for CI/pipeline integration)
- `--dry-run` — show what checks would run without executing them (useful for previewing)
- `--fix` — for simple fixable issues (missing output dirs, missing splits), offer to fix them automatically

---

## Implementation Order

### Phase 1: Foundation + Level 1 (Preflight)

| Step | Task | Depends On |
|------|------|-----------|
| 1.0 | Project scaffolding: pyproject.toml, package structure, CLI skeleton | — |
| 1.1 | `knowledge/rules.py` — hyperparameter rule definitions | — |
| 1.2 | `preflight/environment.py` — environment checks | — |
| 1.3 | `preflight/data.py` — data validation | — |
| 1.4 | `preflight/config.py` — config validation | 1.1 |
| 1.5 | `preflight/paths.py` — path validation | — |
| 1.6 | `preflight/resources.py` — VRAM/time estimation | 1.1 |
| 1.7 | `preflight/report.py` — report generation | 1.2-1.6 |
| 1.8 | `cli.py` — wire up `--setup` command | 1.7 |
| 1.9 | Tests for Level 1 | 1.2-1.8 |

### Phase 2: Level 2 (Design)

| Step | Task | Depends On |
|------|------|-----------|
| 2.0 | `knowledge/models.py` — model database | — |
| 2.1 | `design/interview.py` — interactive interview | — |
| 2.2 | `design/architecture.py` — recommendation engine | 2.0, 1.1 |
| 2.3 | `design/heuristics.py` — dimension/schedule heuristics | 1.1 |
| 2.4 | `design/spec.py` — spec + report generation | 2.2, 2.3 |
| 2.5 | `cli.py` — wire up `--design` command | 2.4 |
| 2.6 | Tests for Level 2 | 2.1-2.5 |

### Phase 3: Level 3 (Data Strategy)

| Step | Task | Depends On |
|------|------|-----------|
| 3.0 | `knowledge/failures.py` — failure mode catalog | — |
| 3.1 | `data_strategy/analyzer.py` — quality metrics | — |
| 3.2 | `data_strategy/strategy.py` — strategy recommender | 2.0 |
| 3.3 | `data_strategy/sources.py` — source suggestions | — |
| 3.4 | `cli.py` — wire up `--data` command | 3.1-3.3 |
| 3.5 | Tests for Level 3 | 3.1-3.4 |

### Phase 4: Integration

| Step | Task | Depends On |
|------|------|-----------|
| 4.0 | `--full` command (design → data → setup pipeline) | Phases 1-3 |
| 4.1 | TransXform integration — pass config + spec to TransXform | Phase 1 |
| 4.2 | Non-interactive mode for CI/automation | Phases 1-3 |

---

## Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "typer[all]>=0.9",       # CLI framework
    "rich>=13.0",            # Terminal formatting
    "pyyaml>=6.0",           # Config parsing
    "pydantic>=2.0",         # Data validation / dataclasses
]

[project.optional-dependencies]
ml = [
    "torch>=2.0",            # GPU/VRAM introspection
    "transformers>=4.30",    # Model config loading
    "datasets>=2.14",        # Dataset loading
    "tokenizers>=0.13",      # Tokenization for seq length checks
]
analysis = [
    "nltk>=3.8",             # BLEU computation for self-BLEU
    "datasketch>=1.6",       # MinHash for dedup
]
```

Core TransXLab should install without ML dependencies (for CI, dry-run config validation). ML features activate when the optional `ml` extras are installed.

---

## Open Questions

1. **Config format standardization:** Should TransXLab define its own config schema, or adapt to common formats (HuggingFace TrainingArguments, axolotl, etc.)?
   - *Recommendation:* Define our own minimal schema, with importers for HF/axolotl formats.

2. **How opinionated should recommendations be?** A wrong recommendation is worse than no recommendation.
   - *Recommendation:* Always show confidence level. For low confidence, present options instead of a single recommendation.

3. **TransXform integration depth:** Should TransXLab emit a TransXform spec file directly?
   - *Recommendation:* Yes. TransXLab's config output should include a `transxform_spec` section that TransXform can consume. This closes the loop.

4. **LLM-assisted analysis:** Should Level 3's data quality analysis optionally call an LLM to assess content quality (not just statistical metrics)?
   - *Recommendation:* Defer. Keep it statistical for v1. LLM-assisted quality checks are a good v2 feature.

---

## Success Criteria

| Level | Success Looks Like |
|-------|-------------------|
| Level 1 | Run `transxlab --setup` on an existing training config. It catches at least the lr and VRAM issues from the AC postmortem. Preflight report is clear and actionable. |
| Level 2 | Run `transxlab --design`, answer the interview, get a design spec that a competent ML engineer would agree with for common cases. Recommendations include rationale. |
| Level 3 | Run `transxlab --data` on the AC training data. It flags high self-BLEU and template contamination risk. Data plan suggests distillation as the strategy. |
| Integration | Run `transxlab --full` end-to-end. Design → data plan → config → preflight report → ready to train with TransXform. |
