# TransXLab Examples

Worked examples showing TransXLab in action. Each includes sample data, design inputs, and training configs.

## Quick Start

```bash
# Run any example with the full pipeline:
transxlab full -a examples/01_sentiment_classifier/answers.yaml \
               -d examples/01_sentiment_classifier/data \
               -c examples/01_sentiment_classifier/config.yaml

# Or just the design phase:
transxlab design -a examples/01_sentiment_classifier/answers.yaml
```

## Examples

### 01: Sentiment Classifier
- **Task:** Classify customer reviews as positive/negative/neutral
- **Model:** Gemma-2-2B with LoRA
- **VRAM:** 24GB (RTX 3090/4090)
- **Demonstrates:** Classification task, small LoRA rank, consistency priority

### 02: Code Assistant
- **Task:** Generate code from natural language instructions
- **Model:** Qwen-2.5-7B with QLoRA
- **VRAM:** 24GB
- **Demonstrates:** Generation task, QLoRA 4-bit quantization, longer sequences

### 03: Creative Writer
- **Task:** Generate creative story continuations
- **Model:** Mistral-7B-Instruct with LoRA
- **VRAM:** 24GB
- **Demonstrates:** High creativity priority, diversity checks, self-BLEU analysis

### 04: Bad Config Demo (AC-v2 Postmortem)
- **Task:** Relational knowledge generation with intentionally bad config
- **Model:** Llama-3-8B with FULL fine-tune (wrong choice)
- **VRAM:** 24GB (will be flagged as insufficient)
- **Demonstrates:** TransXLab catching multiple issues:
  - Learning rate too high for fine-tuning (1e-4 vs recommended 3e-5)
  - Too many epochs (10 vs recommended 3-5)
  - Full fine-tune requires more VRAM than available
  - Missing diversity loss for creative generation task
  - Templated training data (high self-BLEU)

## Output

Each run produces a `transxlab_output/` directory containing:
- `preflight_report.md` — Environment and config validation
- `design_report.md` — Architecture recommendations
- `design_spec.yaml` — TransXform handoff spec
- `data_plan.md` — Data quality analysis and strategy
- `cost_estimate.md` — Cloud GPU cost estimates
- `configs/` — Ready-to-use training configs:
  - `hf_trainer.yaml` — HuggingFace Trainer
  - `axolotl.yaml` — Axolotl
  - `llamafactory.yaml` — LLaMA-Factory
  - `peft_lora.yaml` — PEFT LoRA config (if applicable)
