# TransXLab

**The training architect: validate and design LLM fine-tuning configs before you spend a dollar on GPU time.**

<!-- Logo excluded from crate; see TransXlab logo.png in repo root -->

3.3 MB single binary. Zero Python dependencies. Catches the mistakes that cost you $665 and a weekend.

[Landing Page](https://zamfir70.github.io/transxlab/) | [Blog: The $665 Postmortem](https://zamfir70.github.io/transxlab/blog-ac-v2.html) | [TransXform (training supervisor)](https://github.com/zamfir70/TransXform)

---

## Install

```
cargo install transxlab
```

## Quick Start

```bash
# Full pipeline: preflight + design + data strategy
transxlab full --config run.yaml

# Validate environment and config only
transxlab setup --config run.yaml

# Design architecture from a HuggingFace model
transxlab design --inputs design.yaml --hub-id meta-llama/Llama-3-8B

# Postmortem on a failed run
transxlab diagnose --log training.log
```

## What It Does

TransXLab runs a three-level pipeline over your training config:

| Level | Stage | Catches |
|-------|-------|---------|
| 1 | **Preflight** | Bad env, missing files, GPU/VRAM mismatches, config errors |
| 2 | **Design** | Wrong architecture choices, unsafe hyperparameters, cost blowouts |
| 3 | **Data Strategy** | Quality gaps, contamination risk, diversity issues |

Under the hood: **20 failure-mode signatures**, **25 hyperparameter rules**, cloud cost estimation across 7 GPU tiers and 4 providers (RunPod, Lambda, AWS, Vast.ai).

**HuggingFace Hub integration** -- pass a model ID and TransXLab auto-detects architecture, parameter count, and recommended PEFT config.

**Config generation** for HF Trainer, Axolotl, LLaMA-Factory, and PEFT -- validated configs you can hand straight to your training framework.

**CI/CD gating** with `--fail-on warn|fail` and `--json` for machine-readable output.

## Example: Catching the AC-v2 Disaster

The config that started this project -- a Flan-T5-XL creative-generation run that burned $665 before anyone noticed the problems:

```
$ transxlab full --config examples/ac_v2_config.yaml

TransXLab v0.1.0 -- Full Pipeline
==================================

[PREFLIGHT]  6 checks passed, 0 warnings, 0 failures

[DESIGN]
  FAIL  lr=1e-4 exceeds safe threshold for full fine-tuning (max 5e-5)
  WARN  full fine-tuning on 3B params -- consider LoRA/QLoRA to cut VRAM 60-80%
  WARN  no diversity loss for creative-generation task

[DATA STRATEGY]
  WARN  diversity_loss_weight=0.0 with task_type=creative generation
        --> high mode-collapse risk

[COST]
  Estimated: $142-$310 across providers (A100 80GB recommended)

Result: 2 failures, 2 warnings -- BLOCKED
```

Every issue flagged here went undetected in the real AC-v2 run. TransXLab exists so it doesn't happen again.

## Companion: TransXform

TransXLab validates **before** training. [TransXform](https://github.com/zamfir70/TransXform) supervises **during** training -- detecting loss anomalies, checkpoint corruption, and resource exhaustion in real time.

## License

MIT
