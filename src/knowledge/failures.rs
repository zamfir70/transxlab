/// Known failure modes with detection signatures.

#[derive(Debug, Clone)]
pub struct FailureMode {
    pub name: &'static str,
    pub detection_phase: &'static str,
    pub signals: &'static [&'static str],
    pub consequence: &'static str,
    pub mitigation: &'static str,
    pub source: &'static str,
}

pub static FAILURE_MODES: &[FailureMode] = &[
    FailureMode {
        name: "LR Too High (Fine-tune)",
        detection_phase: "preflight",
        signals: &["lr > 5e-5 for fine-tune context"],
        consequence: "Catastrophic forgetting, loss divergence.",
        mitigation: "Use lr in range 1e-5 to 5e-5 for fine-tuning pretrained models.",
        source: "AC-v2-postmortem",
    },
    FailureMode {
        name: "Template Memorization",
        detection_phase: "preflight",
        signals: &["self_bleu > 0.6 in training data outputs"],
        consequence: "Model reproduces templates instead of learning real generation.",
        mitigation: "Reduce self-BLEU via distillation with diverse prompts. Avoid NLI-format data.",
        source: "AC-v2-postmortem",
    },
    FailureMode {
        name: "Missing Diversity Signal",
        detection_phase: "preflight",
        signals: &["diversity_loss_weight == 0 for creative generation task"],
        consequence: "Mode collapse, repetitive outputs.",
        mitigation: "Set diversity_loss_weight >= 0.3 for creative generation tasks.",
        source: "AC-v2-postmortem",
    },
    FailureMode {
        name: "Train/Val Overlap",
        detection_phase: "preflight",
        signals: &["set intersection > 0 on input field hashes between train and val"],
        consequence: "Inflated validation metrics, false confidence in model quality.",
        mitigation: "Deduplicate train/val splits. Hash inputs and verify zero intersection.",
        source: "literature",
    },
    FailureMode {
        name: "Warmup Too Long",
        detection_phase: "preflight",
        signals: &["warmup_steps > 0.15 * total_steps"],
        consequence: "Wasted compute, delayed effective learning.",
        mitigation: "Keep warmup to 1-10% of total steps.",
        source: "empirical",
    },
    FailureMode {
        name: "VRAM Overflow",
        detection_phase: "preflight",
        signals: &["estimated VRAM > 95% of available GPU memory"],
        consequence: "OOM crash mid-training, lost progress and compute.",
        mitigation: "Reduce batch size, use gradient accumulation, or switch to LoRA/QLoRA.",
        source: "empirical",
    },
    FailureMode {
        name: "Insufficient Data (Scratch)",
        detection_phase: "preflight",
        signals: &["data_size < 10K for scratch-built model"],
        consequence: "Underfitting, poor generalization.",
        mitigation: "Gather more data, or fine-tune a pretrained model instead of training from scratch.",
        source: "literature",
    },
    FailureMode {
        name: "Wrong Eval Metric",
        detection_phase: "preflight",
        signals: &["generation task with only loss-based evaluation", "no novel-query eval configured"],
        consequence: "No signal on actual output quality. Loss can decrease while outputs degrade.",
        mitigation: "Add generation-quality evaluation on held-out novel queries every N steps.",
        source: "AC-v2-postmortem",
    },
    FailureMode {
        name: "Gradient Explosion",
        detection_phase: "runtime",
        signals: &["grad_norm spikes > 10x moving average", "NaN/Inf in gradients"],
        consequence: "Training instability, loss spikes, possible NaN collapse.",
        mitigation: "Enable gradient clipping (max_grad_norm=1.0). Reduce learning rate. Check for data outliers.",
        source: "empirical",
    },
    FailureMode {
        name: "Mode Collapse",
        detection_phase: "runtime",
        signals: &["pairwise cosine similarity > 0.9 across outputs", "generation diversity drops sharply"],
        consequence: "Model produces near-identical outputs regardless of input.",
        mitigation: "Add diversity loss, reduce lr, increase temperature during eval, check for data imbalance.",
        source: "literature",
    },
    FailureMode {
        name: "Catastrophic Forgetting",
        detection_phase: "runtime",
        signals: &["general benchmark scores drop > 20% during fine-tuning"],
        consequence: "Model loses pretrained capabilities while acquiring new ones.",
        mitigation: "Use LoRA/QLoRA instead of full fine-tune. Lower lr. Add replay buffer of general data.",
        source: "literature",
    },
    FailureMode {
        name: "Data Leakage (Test in Train)",
        detection_phase: "preflight",
        signals: &["test/eval examples found in training set", "suspiciously high val accuracy from epoch 1"],
        consequence: "Overestimated model performance. Fails on real-world inputs.",
        mitigation: "Hash-deduplicate across all splits. Use temporal or domain-based splits.",
        source: "literature",
    },
    FailureMode {
        name: "Batch Size Too Small",
        detection_phase: "preflight",
        signals: &["effective_batch_size < 8 for language model training"],
        consequence: "Noisy gradients, slow convergence, unstable training.",
        mitigation: "Increase batch size or gradient accumulation steps. Effective batch >= 16 recommended.",
        source: "empirical",
    },
    FailureMode {
        name: "Batch Size Too Large",
        detection_phase: "preflight",
        signals: &["effective_batch_size > 512 without lr scaling"],
        consequence: "Sharp minima, poor generalization, wasted compute.",
        mitigation: "Scale lr with sqrt(batch_size). Or reduce batch size and use more steps.",
        source: "literature",
    },
    FailureMode {
        name: "Context Length Exceeded",
        detection_phase: "preflight",
        signals: &["max_seq_len in data > model's max_position_embeddings"],
        consequence: "Silent truncation or OOM. Model never sees full examples.",
        mitigation: "Truncate or chunk data to model's context window. Or choose a model with longer context.",
        source: "empirical",
    },
    FailureMode {
        name: "Label Imbalance",
        detection_phase: "preflight",
        signals: &["class ratio > 10:1 in classification data", "majority class > 90%"],
        consequence: "Model learns to always predict majority class. Accuracy metric is misleading.",
        mitigation: "Use weighted loss, oversample minority, or use F1/balanced accuracy as metric.",
        source: "literature",
    },
    FailureMode {
        name: "Loss Plateau",
        detection_phase: "runtime",
        signals: &["loss change < 0.1% over 500+ steps", "learning rate schedule already decayed"],
        consequence: "Wasting compute with no further learning.",
        mitigation: "Try lr warmup restart, increase lr briefly, add data augmentation, or stop early.",
        source: "empirical",
    },
    FailureMode {
        name: "Checkpoint Gap",
        detection_phase: "preflight",
        signals: &["save_steps > total_steps / 3", "no checkpoint configured"],
        consequence: "Losing hours of training progress on any interruption.",
        mitigation: "Save checkpoints every 10-20% of total steps. Use save_total_limit to manage disk.",
        source: "empirical",
    },
    FailureMode {
        name: "Mixed Precision Instability",
        detection_phase: "runtime",
        signals: &["loss scaler underflows frequently", "NaN only in fp16/bf16 mode"],
        consequence: "Training produces NaN. Gradient scaling fails to recover.",
        mitigation: "Switch to bf16 if GPU supports it. Increase initial loss scale. Add gradient clipping.",
        source: "empirical",
    },
];

pub fn get_preflight_failure_modes() -> Vec<&'static FailureMode> {
    FAILURE_MODES
        .iter()
        .filter(|fm| fm.detection_phase == "preflight")
        .collect()
}

pub fn get_runtime_failure_modes() -> Vec<&'static FailureMode> {
    FAILURE_MODES
        .iter()
        .filter(|fm| fm.detection_phase == "runtime")
        .collect()
}
