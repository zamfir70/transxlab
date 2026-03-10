/// Config validation: hyperparameter sanity checks.
use crate::knowledge::lessons::find_lesson_for_rule;
use crate::knowledge::rules::{get_rules_for_context, Severity, TrainingContext};
use crate::preflight::environment::CheckResult;
use serde_yaml::Value;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct ConfigReport {
    pub checks: Vec<CheckResult>,
    pub config: HashMap<String, Value>,
}

impl ConfigReport {
    pub fn worst_severity(&self) -> Severity {
        if self.checks.iter().any(|c| c.status == Severity::Fail) {
            Severity::Fail
        } else if self.checks.iter().any(|c| c.status == Severity::Warn) {
            Severity::Warn
        } else {
            Severity::Info
        }
    }

    fn add(&mut self, name: &str, status: Severity, message: &str, detail: &str) {
        self.checks.push(CheckResult {
            name: name.to_string(),
            status,
            message: message.to_string(),
            detail: detail.to_string(),
        });
    }
}

/// Detect training context from config.
pub fn detect_context(flat: &HashMap<String, f64>, raw: &HashMap<String, Value>) -> TrainingContext {
    let method = raw
        .get("method")
        .or_else(|| raw.get("training_method"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    if method.contains("qlora") {
        return TrainingContext::Qlora;
    }
    if method.contains("lora") {
        return TrainingContext::Lora;
    }
    if flat.contains_key("lora_r") || flat.contains_key("lora_rank") {
        let has_quant = raw.get("quantization").is_some()
            || raw.get("load_in_4bit").and_then(|v| v.as_bool()).unwrap_or(false);
        return if has_quant {
            TrainingContext::Qlora
        } else {
            TrainingContext::Lora
        };
    }
    let scratch = raw
        .get("from_scratch")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    if ["true", "1", "yes"].contains(&scratch.as_str()) {
        return TrainingContext::Scratch;
    }
    TrainingContext::FineTune
}

/// Load a YAML config and return (raw map, flattened numeric map).
pub fn load_config(
    path: &std::path::Path,
) -> crate::error::Result<(HashMap<String, Value>, HashMap<String, f64>)> {
    let text = std::fs::read_to_string(path)?;
    let val: Value = serde_yaml::from_str(&text)?;
    let raw = match val {
        Value::Mapping(m) => m
            .into_iter()
            .filter_map(|(k, v)| k.as_str().map(|s| (s.to_string(), v)))
            .collect(),
        _ => HashMap::new(),
    };
    let flat = flatten_numeric(&raw);
    Ok((raw, flat))
}

fn flatten_numeric(raw: &HashMap<String, Value>) -> HashMap<String, f64> {
    let mut flat = HashMap::new();
    for (k, v) in raw {
        if let Some(n) = value_to_f64(v) {
            flat.insert(k.clone(), n);
        }
        // Also handle nested "training.lr" style
        if let Value::Mapping(m) = v {
            for (sk, sv) in m {
                if let (Some(sk_str), Some(n)) = (sk.as_str(), value_to_f64(sv)) {
                    let full = format!("{k}.{sk_str}");
                    flat.insert(full, n);
                    // Short alias
                    if !flat.contains_key(sk_str) {
                        flat.insert(sk_str.to_string(), n);
                    }
                }
            }
        }
    }
    flat
}

fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Run config validation.
pub fn check_config(
    raw: &HashMap<String, Value>,
    flat: &HashMap<String, f64>,
    n_examples: usize,
) -> ConfigReport {
    let mut report = ConfigReport {
        config: raw.clone(),
        ..Default::default()
    };

    let context = detect_context(flat, raw);
    report.add("Training context", Severity::Info, &format!("Detected: {context}"), "");

    // Check hyperparams
    let rules = get_rules_for_context(context);
    for rule in &rules {
        let value = flat.get(rule.parameter);
        let value = value.copied();
        if let Some(val) = value {
            if let Some((sev, msg)) = rule.check(val) {
                // Enrich with lesson
                let mut full_msg = msg;
                let failure_name = match (rule.parameter, context) {
                    ("lr", TrainingContext::FineTune) => "LR Too High (Fine-tune)",
                    _ => "",
                };
                if let Some(lesson) = find_lesson_for_rule(failure_name) {
                    let finding = lesson
                        .findings
                        .iter()
                        .find(|f| f.key == rule.parameter)
                        .map(|f| f.message)
                        .unwrap_or(lesson.summary);
                    full_msg = format!("{full_msg} See lesson {}: '{finding}'", lesson.id);
                }
                report.add(&format!("Param: {}", rule.parameter), sev, &full_msg, "");
            } else {
                report.add(
                    &format!("Param: {}", rule.parameter),
                    Severity::Info,
                    &format!("{}={:.1e} is within sane range for {context}", rule.parameter, val),
                    "",
                );
            }
        }
    }

    // Warmup ratio
    if let Some(warmup) = flat.get("warmup_steps").or_else(|| flat.get("warmup")) {
        let warmup = *warmup as usize;
        if n_examples > 0 {
            let batch = flat.get("batch_size").or_else(|| flat.get("per_device_train_batch_size")).copied().unwrap_or(1.0) as usize;
            let accum = flat.get("grad_accum_steps").or_else(|| flat.get("gradient_accumulation_steps")).copied().unwrap_or(1.0) as usize;
            let epochs = flat.get("epochs").or_else(|| flat.get("num_train_epochs")).copied().unwrap_or(1.0) as usize;
            let eff = batch * accum;
            let spe = n_examples.div_ceil(eff);
            let total = spe * epochs;
            if total > 0 {
                let ratio = warmup as f64 / total as f64;
                if ratio > 0.15 {
                    report.add(
                        "Warmup ratio",
                        Severity::Warn,
                        &format!("Warmup {warmup} steps is {:.0}% of {total} total steps. Recommend 1-10%.", ratio * 100.0),
                        "",
                    );
                } else {
                    report.add(
                        "Warmup ratio",
                        Severity::Info,
                        &format!("Warmup {warmup}/{total} steps ({:.0}%)", ratio * 100.0),
                        "OK",
                    );
                }
            }
        }
    }

    // LoRA alpha/r ratio
    if context == TrainingContext::Lora || context == TrainingContext::Qlora {
        let r = flat.get("lora_r").or_else(|| flat.get("lora_rank")).copied();
        let alpha = flat.get("lora_alpha").copied();
        if let (Some(r), Some(alpha)) = (r, alpha) {
            let ratio = alpha / r;
            if !(1.0..=4.0).contains(&ratio) {
                report.add(
                    "LoRA alpha/r ratio",
                    Severity::Warn,
                    &format!("lora_alpha/lora_r = {alpha:.0}/{r:.0} = {ratio:.1}. Typical range is 1-4 (convention: alpha = 2*r)."),
                    "",
                );
            } else {
                report.add("LoRA alpha/r ratio", Severity::Info, &format!("lora_alpha/lora_r = {ratio:.1}"), "OK");
            }
        }
    }

    // Diversity signal
    let task_type = raw
        .get("task_type")
        .or_else(|| raw.get("task"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    if task_type.contains("creative") || task_type.contains("generation") {
        if let Some(dw) = flat.get("diversity_loss_weight") {
            if *dw == 0.0 {
                let detail = find_lesson_for_rule("Missing Diversity Signal")
                    .map(|l| {
                        let msg = l
                            .findings
                            .iter()
                            .find(|f| f.key == "diversity_loss")
                            .map(|f| f.message)
                            .unwrap_or("");
                        format!("See lesson {}: '{msg}'", l.id)
                    })
                    .unwrap_or_default();
                report.add(
                    "Diversity signal",
                    Severity::Warn,
                    "diversity_loss_weight=0.0 for a creative/generation task. Risk of mode collapse.",
                    &detail,
                );
            }
        }
    }

    report
}
