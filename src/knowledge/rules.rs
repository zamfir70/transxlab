/// Declarative hyperparameter rules and data quality thresholds.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Fail,
    Warn,
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Fail => write!(f, "fail"),
            Severity::Warn => write!(f, "warn"),
            Severity::Info => write!(f, "info"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingContext {
    FineTune,
    Lora,
    Qlora,
    Scratch,
    Any,
}

impl std::fmt::Display for TrainingContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingContext::FineTune => write!(f, "fine-tune"),
            TrainingContext::Lora => write!(f, "lora"),
            TrainingContext::Qlora => write!(f, "qlora"),
            TrainingContext::Scratch => write!(f, "scratch"),
            TrainingContext::Any => write!(f, "any"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HyperparamRule {
    pub parameter: &'static str,
    pub context: TrainingContext,
    pub min_val: Option<f64>,
    pub max_val: Option<f64>,
    pub recommended: Option<f64>,
    pub severity: Severity,
    pub rationale: &'static str,
    pub source: &'static str,
}

impl HyperparamRule {
    /// Check a value against this rule. Returns (severity, message) if violated.
    pub fn check(&self, value: f64) -> Option<(Severity, String)> {
        if let Some(min) = self.min_val {
            if value < min {
                return Some((
                    self.severity,
                    format!(
                        "{}={:.1e} is below minimum {:.1e} for {}. {} [source: {}]",
                        self.parameter, value, min, self.context, self.rationale, self.source,
                    ),
                ));
            }
        }
        if let Some(max) = self.max_val {
            if value > max {
                return Some((
                    self.severity,
                    format!(
                        "{}={:.1e} is above maximum {:.1e} for {}. {} [source: {}]",
                        self.parameter, value, max, self.context, self.rationale, self.source,
                    ),
                ));
            }
        }
        None
    }
}

pub static HYPERPARAM_RULES: &[HyperparamRule] = &[
    // Learning rate
    HyperparamRule {
        parameter: "lr",
        context: TrainingContext::FineTune,
        min_val: Some(1e-6),
        max_val: Some(5e-5),
        recommended: Some(3e-5),
        severity: Severity::Warn,
        rationale: "Higher lr risks catastrophic forgetting on pretrained weights.",
        source: "AC-v2-postmortem",
    },
    HyperparamRule {
        parameter: "lr",
        context: TrainingContext::Lora,
        min_val: Some(1e-5),
        max_val: Some(5e-4),
        recommended: Some(2e-4),
        severity: Severity::Warn,
        rationale: "LoRA adapters tolerate slightly higher lr, but beyond 5e-4 is unusual.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "lr",
        context: TrainingContext::Qlora,
        min_val: Some(1e-5),
        max_val: Some(5e-4),
        recommended: Some(2e-4),
        severity: Severity::Warn,
        rationale: "QLoRA follows similar lr ranges to LoRA.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "lr",
        context: TrainingContext::Scratch,
        min_val: Some(1e-4),
        max_val: Some(1e-2),
        recommended: Some(1e-3),
        severity: Severity::Warn,
        rationale: "Scratch training needs higher lr for random-init convergence.",
        source: "literature",
    },
    // Epochs
    HyperparamRule {
        parameter: "epochs",
        context: TrainingContext::FineTune,
        min_val: Some(1.0),
        max_val: Some(5.0),
        recommended: Some(3.0),
        severity: Severity::Warn,
        rationale: "More than 5 epochs on fine-tuning risks overfitting, especially on small data.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "epochs",
        context: TrainingContext::Lora,
        min_val: Some(1.0),
        max_val: Some(10.0),
        recommended: Some(3.0),
        severity: Severity::Warn,
        rationale: "LoRA has fewer trainable params, can tolerate more epochs, but >10 is unusual.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "epochs",
        context: TrainingContext::Scratch,
        min_val: Some(3.0),
        max_val: Some(50.0),
        recommended: Some(10.0),
        severity: Severity::Info,
        rationale: "Scratch training epoch count depends heavily on dataset size.",
        source: "literature",
    },
    // Weight decay
    HyperparamRule {
        parameter: "weight_decay",
        context: TrainingContext::Any,
        min_val: Some(0.0),
        max_val: Some(0.1),
        recommended: Some(0.01),
        severity: Severity::Warn,
        rationale: "Weight decay > 0.1 is unusual and may hurt convergence.",
        source: "literature",
    },
    // Gradient accumulation
    HyperparamRule {
        parameter: "grad_accum_steps",
        context: TrainingContext::Any,
        min_val: Some(1.0),
        max_val: Some(64.0),
        recommended: Some(4.0),
        severity: Severity::Warn,
        rationale: "Effective batch = batch_size * grad_accum. >64 accumulation steps is unusual.",
        source: "empirical",
    },
    // LoRA rank
    HyperparamRule {
        parameter: "lora_r",
        context: TrainingContext::Lora,
        min_val: Some(4.0),
        max_val: Some(64.0),
        recommended: Some(16.0),
        severity: Severity::Warn,
        rationale: "r<4 has too little capacity; r>64 approaches full fine-tune cost.",
        source: "literature",
    },
    HyperparamRule {
        parameter: "lora_r",
        context: TrainingContext::Qlora,
        min_val: Some(4.0),
        max_val: Some(64.0),
        recommended: Some(16.0),
        severity: Severity::Warn,
        rationale: "r<4 has too little capacity; r>64 approaches full fine-tune cost.",
        source: "literature",
    },
    // Dropout
    HyperparamRule {
        parameter: "dropout",
        context: TrainingContext::Any,
        min_val: Some(0.0),
        max_val: Some(0.5),
        recommended: Some(0.1),
        severity: Severity::Warn,
        rationale: "Dropout > 0.5 aggressively suppresses learning.",
        source: "literature",
    },
    // Label smoothing
    HyperparamRule {
        parameter: "label_smoothing",
        context: TrainingContext::Any,
        min_val: Some(0.0),
        max_val: Some(0.2),
        recommended: Some(0.1),
        severity: Severity::Warn,
        rationale: "Label smoothing > 0.2 significantly degrades sharp predictions.",
        source: "literature",
    },
    // Warmup ratio
    HyperparamRule {
        parameter: "warmup_ratio",
        context: TrainingContext::FineTune,
        min_val: Some(0.01),
        max_val: Some(0.1),
        recommended: Some(0.06),
        severity: Severity::Warn,
        rationale: "Warmup > 10% wastes compute; < 1% causes lr spikes on cold weights.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "warmup_ratio",
        context: TrainingContext::Lora,
        min_val: Some(0.01),
        max_val: Some(0.1),
        recommended: Some(0.03),
        severity: Severity::Warn,
        rationale: "LoRA adapters initialize near zero, brief warmup is sufficient.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "warmup_ratio",
        context: TrainingContext::Scratch,
        min_val: Some(0.01),
        max_val: Some(0.15),
        recommended: Some(0.05),
        severity: Severity::Info,
        rationale: "Random-init models benefit from moderate warmup to stabilize early training.",
        source: "literature",
    },
    // Max gradient norm
    HyperparamRule {
        parameter: "max_grad_norm",
        context: TrainingContext::Any,
        min_val: Some(0.1),
        max_val: Some(10.0),
        recommended: Some(1.0),
        severity: Severity::Warn,
        rationale: "Gradient clipping outside [0.1, 10.0] is unusual. 1.0 is the standard default.",
        source: "literature",
    },
    // Batch size (per device)
    HyperparamRule {
        parameter: "per_device_train_batch_size",
        context: TrainingContext::FineTune,
        min_val: Some(1.0),
        max_val: Some(64.0),
        recommended: Some(8.0),
        severity: Severity::Warn,
        rationale: "Batch > 64 per device is unusual for fine-tuning and wastes VRAM.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "per_device_train_batch_size",
        context: TrainingContext::Lora,
        min_val: Some(1.0),
        max_val: Some(32.0),
        recommended: Some(4.0),
        severity: Severity::Warn,
        rationale: "LoRA training usually uses smaller batches since adapters are cheap.",
        source: "empirical",
    },
    // LoRA alpha
    HyperparamRule {
        parameter: "lora_alpha",
        context: TrainingContext::Lora,
        min_val: Some(8.0),
        max_val: Some(128.0),
        recommended: Some(32.0),
        severity: Severity::Warn,
        rationale: "lora_alpha scales adapter impact. Usually set to 2*lora_r or 32.",
        source: "literature",
    },
    HyperparamRule {
        parameter: "lora_alpha",
        context: TrainingContext::Qlora,
        min_val: Some(8.0),
        max_val: Some(128.0),
        recommended: Some(32.0),
        severity: Severity::Warn,
        rationale: "lora_alpha scales adapter impact. Usually set to 2*lora_r or 32.",
        source: "literature",
    },
    // LoRA dropout
    HyperparamRule {
        parameter: "lora_dropout",
        context: TrainingContext::Lora,
        min_val: Some(0.0),
        max_val: Some(0.3),
        recommended: Some(0.05),
        severity: Severity::Warn,
        rationale: "LoRA dropout > 0.3 is excessive given the low-rank bottleneck already regularizes.",
        source: "empirical",
    },
    HyperparamRule {
        parameter: "lora_dropout",
        context: TrainingContext::Qlora,
        min_val: Some(0.0),
        max_val: Some(0.3),
        recommended: Some(0.05),
        severity: Severity::Warn,
        rationale: "LoRA dropout > 0.3 is excessive given the low-rank bottleneck already regularizes.",
        source: "empirical",
    },
    // Max sequence length
    HyperparamRule {
        parameter: "max_seq_len",
        context: TrainingContext::Any,
        min_val: Some(32.0),
        max_val: Some(131072.0),
        recommended: None,
        severity: Severity::Warn,
        rationale: "Sequence length < 32 is unusually short. > 128K requires RoPE-extended models.",
        source: "empirical",
    },
    // Save steps ratio (save_steps / total_steps)
    HyperparamRule {
        parameter: "save_steps_ratio",
        context: TrainingContext::Any,
        min_val: Some(0.02),
        max_val: Some(0.5),
        recommended: Some(0.1),
        severity: Severity::Warn,
        rationale: "Save too rarely and you lose progress on crash. Too often wastes disk I/O.",
        source: "empirical",
    },
    // Diversity loss weight (for generation tasks)
    HyperparamRule {
        parameter: "diversity_loss_weight",
        context: TrainingContext::FineTune,
        min_val: Some(0.0),
        max_val: Some(1.0),
        recommended: Some(0.3),
        severity: Severity::Info,
        rationale: "For creative generation, diversity_loss_weight > 0 prevents mode collapse.",
        source: "AC-v2-postmortem",
    },
];

/// Return all rules applicable to a training context.
pub fn get_rules_for_context(context: TrainingContext) -> Vec<&'static HyperparamRule> {
    HYPERPARAM_RULES
        .iter()
        .filter(|r| r.context == context || r.context == TrainingContext::Any)
        .collect()
}

// ---------------------------------------------------------------------------
// Scratch-build dimension heuristics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ScratchDimensions {
    pub tier: &'static str,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_encoder_layers: u32,
    pub n_decoder_layers: u32,
    pub d_ff: u32,
    pub approx_params: &'static str,
}

pub static SCRATCH_HEURISTICS: &[ScratchDimensions] = &[
    ScratchDimensions { tier: "tiny",   d_model: 256,  n_heads: 4,  n_encoder_layers: 4,  n_decoder_layers: 4,  d_ff: 1024, approx_params: "~25M" },
    ScratchDimensions { tier: "small",  d_model: 512,  n_heads: 8,  n_encoder_layers: 6,  n_decoder_layers: 6,  d_ff: 2048, approx_params: "~125M" },
    ScratchDimensions { tier: "medium", d_model: 768,  n_heads: 12, n_encoder_layers: 12, n_decoder_layers: 12, d_ff: 3072, approx_params: "~350M" },
    ScratchDimensions { tier: "large",  d_model: 1024, n_heads: 16, n_encoder_layers: 24, n_decoder_layers: 24, d_ff: 4096, approx_params: "~770M" },
];

// ---------------------------------------------------------------------------
// VRAM estimation constants
// ---------------------------------------------------------------------------

/// Bytes per parameter for each precision.
pub fn bytes_per_param(precision: &str) -> f64 {
    match precision {
        "fp32" => 4.0,
        "fp16" | "bf16" => 2.0,
        "int8" => 1.0,
        "int4" => 0.5,
        _ => 2.0,
    }
}

/// AdamW stores 2 fp32 states per trainable param.
pub const ADAMW_STATE_BYTES_PER_PARAM: f64 = 8.0;
