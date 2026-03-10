# TransXLab Next: Feature Checklist

Priority-ordered. Check off as completed.

## P1: `transxlab full` — unified pipeline
- [x] `cmd_full` in CLI: runs setup → design → data in sequence
- [x] Accepts superset of all flags: `--config`, `--answers`, `--data-dir`, `--output-dir`, etc.
- [x] Single unified report (markdown + JSON) combining all three levels
- [x] `--fail-on <warn|fail>` exit code control for CI gating
- [x] Emits `transxform_spec.yaml` as final artifact

## P2: CI/non-interactive mode with exit codes
- [x] `--fail-on warn` exits non-zero on any warning (default: only on fail)
- [x] `--fail-on fail` exits non-zero only on hard failures
- [x] Machine-readable JSON summary to stdout (separate from stderr reports)
- [x] Works with `transxlab full` and `transxlab setup`

## P3: `transxlab diagnose` — postmortem mode
- [x] Accepts `--log <path>` (training log file, JSON-lines or plain text)
- [x] Parse loss curves, metric trajectories from log
- [x] Match against failure mode signatures (template memorization, mode collapse, etc.)
- [x] Emit diagnosis report with matched patterns, confidence, and remediation
- [x] Reference specific failure modes and lessons from knowledge base

## P4: Knowledge base expansion
- [x] More failure modes beyond the initial 8 (now 20: added gradient explosion, mode collapse, catastrophic forgetting, data leakage, batch size, context length, label imbalance, loss plateau, checkpoint gap, mixed precision instability)
- [x] More model entries (added Gemma-2-9B/2B, Qwen-2.5-14B/3B, DeepSeek-V2-Lite, Llama-3.1-70B, Phi-3-Medium, Mistral-Nemo-12B)
- [x] More hyperparameter rules (added warmup_ratio, max_grad_norm, per_device_train_batch_size, lora_alpha, lora_dropout, max_seq_len, save_steps_ratio, diversity_loss_weight)

## P5: HuggingFace Hub integration
- [x] `--model meta-llama/Llama-3-8B` fetches config.json from HF Hub
- [x] Extract param count, architecture, context length, dtype from remote config
- [x] Falls back to static knowledge base if network unavailable
- [x] No HF token required for public models

## P6: Config generation
- [x] Emit HuggingFace Trainer config (`configs/hf_trainer.yaml`)
- [x] Emit Axolotl config (`configs/axolotl.yaml`)
- [x] Emit LLaMA-Factory config (`configs/llamafactory.yaml`)
- [x] Emit PEFT LoRA config when applicable (`configs/peft_lora.yaml`)
- [x] All configs generated from design recommendations (lr, batch, epochs, LoRA params)

## P7: Cloud cost estimation
- [x] GPU tier database: RTX 3090, RTX 4090, A10G, L40S, A100 40/80GB, H100 80GB
- [x] Provider pricing: RunPod, Lambda, AWS, Vast.ai
- [x] Training time estimation from param count, data size, method (full/LoRA)
- [x] VRAM fit checking per GPU tier
- [x] Best-value recommendation
- [x] Markdown cost report saved to `cost_estimate.md`
- [x] Integrated into `--json` output for CI

## P8: Example gallery
- [x] 01: Sentiment classifier (LoRA, classify, Gemma-2-2B)
- [x] 02: Code assistant (QLoRA, generate, Qwen-2.5-7B)
- [x] 03: Creative writer (LoRA, generate, Mistral-7B, high diversity)
- [x] 04: Bad config demo (AC-v2 postmortem recreation — catches lr, epochs, VRAM, self-BLEU)
- [x] Each example includes answers.yaml, config.yaml, sample data
- [x] README.md with quick-start instructions
