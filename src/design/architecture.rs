/// Architecture recommendation engine: maps design inputs to concrete specs.
use crate::design::heuristics::{
    parse_param_budget, recommend_lora_config, recommend_training_schedule,
    select_scratch_dimensions, LoRAConfig, TrainingSchedule,
};
use crate::design::interview::*;
use crate::knowledge::models::MODELS;
use crate::knowledge::rules::ScratchDimensions;

#[derive(Debug, Clone)]
pub struct Confidence {
    pub level: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct Recommendation {
    pub key: String,
    pub value: String,
    pub rationale: String,
    pub confidence: Confidence,
}

#[derive(Debug, Clone)]
pub struct ArchitectureSpec {
    pub recommendations: Vec<Recommendation>,
    pub base_model: String,
    pub architecture_type: String,
    pub training_method: String,
    pub lora_config: Option<LoRAConfig>,
    pub scratch_dims: Option<&'static ScratchDimensions>,
    pub schedule: Option<TrainingSchedule>,
    pub estimated_params: u64,
    pub estimated_vram_gb: f64,
    pub loss_function: String,
    pub extra_losses: Vec<String>,
}

impl ArchitectureSpec {
    fn new() -> Self {
        Self {
            recommendations: Vec::new(),
            base_model: String::new(),
            architecture_type: String::new(),
            training_method: String::new(),
            lora_config: None,
            scratch_dims: None,
            schedule: None,
            estimated_params: 0,
            estimated_vram_gb: 0.0,
            loss_function: "cross_entropy".to_string(),
            extra_losses: Vec::new(),
        }
    }

    fn add(
        &mut self,
        key: &str,
        value: &str,
        rationale: &str,
        confidence: &str,
        confidence_reason: &str,
    ) {
        self.recommendations.push(Recommendation {
            key: key.to_string(),
            value: value.to_string(),
            rationale: rationale.to_string(),
            confidence: Confidence {
                level: confidence.to_string(),
                reason: confidence_reason.to_string(),
            },
        });
    }
}

pub fn recommend(inputs: &DesignInputs) -> ArchitectureSpec {
    if inputs.approach == TrainingApproach::FineTune {
        recommend_fine_tune(inputs)
    } else {
        recommend_scratch(inputs)
    }
}

fn recommend_fine_tune(inputs: &DesignInputs) -> ArchitectureSpec {
    let mut spec = ArchitectureSpec::new();

    // 1. Base model selection
    let model_lower = inputs.base_model.to_lowercase();
    if model_lower == "recommend" || inputs.base_model.is_empty() {
        if let Some((name, model_spec)) = select_base_model(inputs) {
            spec.base_model = name.to_string();
            spec.add(
                "base_model",
                name,
                &format!(
                    "Selected {} ({} params). {}. VRAM: {} (bf16), {} (QLoRA).",
                    model_spec.name,
                    model_spec.params,
                    model_spec.good_for.join(", "),
                    model_spec.vram_bf16,
                    model_spec.vram_qlora,
                ),
                "medium",
                "Heuristic selection based on task type and VRAM constraint",
            );
            spec.architecture_type = model_spec.architecture.to_string();
            spec.estimated_params = parse_params(model_spec.params);
        }
    } else {
        spec.base_model = inputs.base_model.clone();
        if let Some((_name, model_spec)) = find_model(&inputs.base_model) {
            spec.architecture_type = model_spec.architecture.to_string();
            spec.estimated_params = parse_params(model_spec.params);
            spec.add(
                "base_model",
                &inputs.base_model,
                &format!("User-selected. {} ({} params).", model_spec.name, model_spec.params),
                "high",
                "User explicitly chose this model",
            );
        } else {
            spec.add(
                "base_model",
                &inputs.base_model,
                "User-selected model (not in knowledge base).",
                "low",
                "Model not in database, cannot validate fit",
            );
        }
    }

    // 2. Training method
    let method = inputs.training_method;
    spec.training_method = method.as_str().to_string();
    match method {
        TrainingMethod::Lora => spec.add(
            "method",
            "LoRA",
            "Preserves base model generalization while training only adapter weights.",
            "high",
            "LoRA is well-established for fine-tuning",
        ),
        TrainingMethod::Qlora => spec.add(
            "method",
            "QLoRA",
            "4-bit quantized base with LoRA adapters. Minimal VRAM, some quality trade-off.",
            "high",
            "QLoRA is well-established for memory-constrained fine-tuning",
        ),
        TrainingMethod::Full => spec.add(
            "method",
            "Full fine-tune",
            "All parameters trainable. Best quality but highest VRAM cost.",
            if inputs.vram_gb > 40.0 { "high" } else { "medium" },
            "Full fine-tune needs significant VRAM",
        ),
    }

    // 3. LoRA config
    if method == TrainingMethod::Lora || method == TrainingMethod::Qlora {
        let lora = recommend_lora_config(
            inputs.task_type.as_str(),
            inputs.creativity.as_str(),
            &spec.base_model,
        );
        spec.add(
            "lora_config",
            &format!(
                "r={}, alpha={}, targets={:?}",
                lora.r, lora.alpha, lora.target_modules
            ),
            &format!(
                "Rank {} for {} task. Alpha = 2*r (convention).",
                lora.r,
                if lora.r >= 32 { "creative/generative" } else { "standard" },
            ),
            "medium",
            "LoRA rank selection is task-dependent",
        );
        spec.lora_config = Some(lora);
    }

    // 4. Freeze strategy
    if method == TrainingMethod::Lora || method == TrainingMethod::Qlora {
        spec.add(
            "freeze_strategy",
            "All base weights frozen, train adapters only",
            "Standard for LoRA -- preserves pretrained knowledge.",
            "high",
            "This is how LoRA works by design",
        );
    } else {
        spec.add(
            "freeze_strategy",
            "All weights trainable",
            "Full fine-tune -- all parameters updated.",
            "high",
            "Standard full fine-tuning",
        );
    }

    // 5. Training schedule
    let schedule = recommend_training_schedule(
        "fine-tune",
        method.as_str(),
        inputs.data_size,
        spec.estimated_params,
        inputs.vram_gb,
    );
    spec.add(
        "schedule",
        &format!(
            "lr={}, epochs={}, batch={}x{}={}",
            schedule.lr, schedule.epochs, schedule.batch_size,
            schedule.grad_accum_steps, schedule.effective_batch,
        ),
        &format!(
            "{} lr. Epochs based on {} examples.",
            if method == TrainingMethod::Lora || method == TrainingMethod::Qlora {
                "LoRA-appropriate"
            } else {
                "Fine-tune-appropriate"
            },
            inputs.data_size,
        ),
        "medium",
        "Heuristic schedule, may need tuning",
    );
    spec.schedule = Some(schedule);

    // 6. Loss function
    spec.loss_function = "cross_entropy".to_string();
    if inputs.creativity == CreativityPriority::Creativity
        && inputs.task_type == TaskType::Generate
    {
        spec.extra_losses
            .push("diversity_loss (weight=0.3)".to_string());
        spec.add(
            "loss",
            "cross_entropy + diversity_loss (weight=0.3)",
            "Creative generation benefits from diversity signal to prevent mode collapse.",
            "high",
            "Lesson AC-v2: diversity_loss_weight=0.0 causes mode collapse",
        );
    } else {
        spec.add(
            "loss",
            "cross_entropy",
            "Standard loss for this task.",
            "high",
            "",
        );
    }

    // 7. VRAM estimate
    estimate_vram(&mut spec, inputs);

    spec
}

fn recommend_scratch(inputs: &DesignInputs) -> ArchitectureSpec {
    let mut spec = ArchitectureSpec::new();

    // 1. Architecture family
    let arch_type = select_architecture_family(inputs);
    spec.architecture_type = arch_type.to_string();
    spec.add(
        "architecture",
        arch_type,
        &architecture_rationale(inputs, arch_type),
        if arch_type != "custom" { "high" } else { "low" },
        "Based on task modality and output format",
    );

    // 2. Dimensions
    let dims = select_scratch_dimensions(&inputs.param_budget);
    spec.scratch_dims = Some(dims);
    spec.estimated_params = {
        let s = dims.approx_params.trim_start_matches('~');
        parse_param_budget(s)
    };
    spec.add(
        "dimensions",
        &format!(
            "d_model={}, n_heads={}, layers={}+{}, d_ff={}",
            dims.d_model, dims.n_heads, dims.n_encoder_layers, dims.n_decoder_layers, dims.d_ff,
        ),
        &format!(
            "Tier '{}' ({} params) for budget {}.",
            dims.tier, dims.approx_params, inputs.param_budget,
        ),
        "medium",
        "Standard dimension scaling, actual param count may vary with vocab size",
    );

    // 3. Positional encoding
    let max_seq = std::cmp::max(inputs.input_seq_len, inputs.output_seq_len);
    let pos_enc = if max_seq <= 2048 {
        "learned"
    } else {
        "rotary (RoPE)"
    };
    spec.add(
        "positional_encoding",
        pos_enc,
        &format!(
            "{} for max sequence length {}.",
            if pos_enc == "learned" { "Learned" } else { "RoPE" },
            max_seq,
        ),
        "high",
        "Standard choice for this sequence length",
    );

    // 4. Tokenizer
    spec.add(
        "tokenizer",
        "SentencePiece (vocab_size=32000)",
        "Train SentencePiece on your corpus. 32K vocab balances coverage and embedding size.",
        "medium",
        "Default recommendation, domain-specific corpus may need adjustment",
    );

    // 5. Normalization
    spec.add(
        "normalization",
        "pre-norm (RMSNorm)",
        "Pre-norm is more stable during training than post-norm.",
        "high",
        "Pre-norm is the modern standard",
    );

    // 6. Training method and schedule
    spec.training_method = "scratch".to_string();
    let schedule = recommend_training_schedule(
        "scratch",
        "full",
        inputs.data_size,
        spec.estimated_params,
        inputs.vram_gb,
    );
    spec.add(
        "schedule",
        &format!(
            "lr={}, epochs={}, batch={}x{}={}, warmup={}",
            schedule.lr, schedule.epochs, schedule.batch_size,
            schedule.grad_accum_steps, schedule.effective_batch, schedule.warmup_steps,
        ),
        &format!("Scratch training schedule for {} examples.", inputs.data_size),
        "medium",
        "Heuristic schedule, calibrate with initial training runs",
    );
    spec.schedule = Some(schedule);

    // 7. Loss
    spec.loss_function = "cross_entropy".to_string();
    if inputs.task_type == TaskType::Generate {
        spec.extra_losses
            .push("diversity_loss (weight=0.3)".to_string());
        spec.add(
            "loss",
            "cross_entropy + diversity_loss",
            "Generation task benefits from diversity signal.",
            "medium",
            "Recommended for generation tasks",
        );
    } else {
        spec.add(
            "loss",
            "cross_entropy",
            "Standard loss.",
            "high",
            "",
        );
    }

    // 8. Data viability check
    if inputs.data_size > 0 && inputs.data_size < 10000 {
        spec.add(
            "data_warning",
            &format!(
                "{} examples may be insufficient for scratch training",
                inputs.data_size,
            ),
            "Consider fine-tuning a pretrained model instead, or gathering more data.",
            "high",
            "Literature consensus: scratch training needs >10K examples",
        );
    }

    // 9. VRAM estimate
    estimate_vram(&mut spec, inputs);

    spec
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn select_base_model(inputs: &DesignInputs) -> Option<(&'static str, &'static crate::knowledge::models::ModelSpec)> {
    let mut candidates: Vec<(&str, &crate::knowledge::models::ModelSpec, f64)> = Vec::new();

    for (name, model) in MODELS.iter() {
        let mut score: f64 = 0.0;

        // Task fit
        let task_str = inputs.task_type.as_str();
        if model.good_for.iter().any(|g| g.contains(task_str)) {
            score += 10.0;
        }

        // VRAM fit
        let model_vram = if inputs.training_method == TrainingMethod::Qlora {
            parse_vram(model.vram_qlora)
        } else {
            parse_vram(model.vram_bf16)
        };

        if model_vram <= inputs.vram_gb * 0.7 {
            score += 5.0;
        } else if model_vram <= inputs.vram_gb {
            score += 2.0;
        } else {
            score -= 10.0;
        }

        // Prefer larger models that fit
        let param_count = parse_params(model.params);
        score += (param_count as f64 / 1e9).min(5.0);

        candidates.push((name, model, score));
    }

    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    candidates.first().map(|(name, model, _)| (*name, *model))
}

fn find_model(name: &str) -> Option<(&'static str, &'static crate::knowledge::models::ModelSpec)> {
    let name_lower = name.to_lowercase().replace('-', "").replace('_', "");
    for (key, spec) in MODELS.iter() {
        let key_norm = key.replace('-', "");
        if key_norm.contains(&name_lower) || name_lower.contains(&key_norm) {
            return Some((key, spec));
        }
    }
    None
}

fn select_architecture_family(inputs: &DesignInputs) -> &'static str {
    if inputs.task_type == TaskType::Classify || inputs.task_type == TaskType::Embed {
        return "encoder-only";
    }
    if inputs.output_format == OutputFormat::Text {
        if inputs.input_format == InputFormat::Structured {
            return "encoder-decoder";
        }
        return "decoder-only";
    }
    if inputs.output_format == OutputFormat::Classes {
        return "encoder-only";
    }
    "encoder-decoder"
}

fn architecture_rationale(_inputs: &DesignInputs, arch: &str) -> String {
    match arch {
        "encoder-only" => "Classification/embedding tasks use encoder-only with a task head.".into(),
        "encoder-decoder" => {
            "Structured input to variable-length output is classic encoder-decoder.".into()
        }
        "decoder-only" => {
            "Text generation with text input maps well to decoder-only (autoregressive).".into()
        }
        _ => "Non-standard task -- consider custom architecture.".into(),
    }
}

fn estimate_vram(spec: &mut ArchitectureSpec, inputs: &DesignInputs) {
    let gb = (1024u64 * 1024 * 1024) as f64;
    let params = spec.estimated_params;
    if params == 0 {
        return;
    }

    let method = &spec.training_method;
    if method == "lora" || method == "qlora" {
        let adapter_frac = match &spec.lora_config {
            Some(lora) => (lora.r as f64 / 1600.0).min(0.05),
            None => 0.01,
        };
        let model_mem = if method == "qlora" {
            params as f64 * 0.5 / gb
        } else {
            params as f64 * 2.0 / gb
        };
        let opt_mem = params as f64 * adapter_frac * 12.0 / gb;
        spec.estimated_vram_gb = model_mem + opt_mem + 1.0;
    } else {
        spec.estimated_vram_gb = (params as f64 * 12.0 / gb) + 1.0;
    }

    let fits = spec.estimated_vram_gb <= inputs.vram_gb;
    spec.add(
        "vram_estimate",
        &format!(
            "~{:.1}GB ({} in {:.0}GB)",
            spec.estimated_vram_gb,
            if fits { "fits" } else { "DOES NOT FIT" },
            inputs.vram_gb,
        ),
        if !fits {
            "Reduce batch size or switch to QLoRA if tight."
        } else {
            "Comfortable fit."
        },
        "medium",
        "Rough estimate, actual usage varies with batch size and sequence length",
    );
}

fn parse_params(s: &str) -> u64 {
    let s = s.trim().to_uppercase();
    if let Some(rest) = s.strip_suffix('B') {
        (rest.parse::<f64>().unwrap_or(0.0) * 1e9) as u64
    } else if let Some(rest) = s.strip_suffix('M') {
        (rest.parse::<f64>().unwrap_or(0.0) * 1e6) as u64
    } else {
        s.parse::<u64>().unwrap_or(0)
    }
}

fn parse_vram(s: &str) -> f64 {
    s.to_uppercase()
        .replace("GB", "")
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
}
