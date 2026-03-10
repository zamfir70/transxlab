/// Resource estimation: VRAM, training time, cost.
use crate::knowledge::models::{find_model, parse_param_count};
use crate::knowledge::rules::{bytes_per_param, Severity, ADAMW_STATE_BYTES_PER_PARAM};
use crate::preflight::environment::{CheckResult, get_gpu_vram_bytes};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ResourceEstimates {
    pub total_params: u64,
    pub trainable_params: u64,
    pub model_memory_gb: f64,
    pub optimizer_memory_gb: f64,
    pub gradient_memory_gb: f64,
    pub activation_memory_gb: f64,
    pub total_vram_gb: f64,
    pub steps_per_epoch: usize,
    pub total_steps: usize,
    pub estimated_time_minutes: f64,
    pub estimated_cost_usd: f64,
}

#[derive(Debug, Default)]
pub struct ResourceReport {
    pub checks: Vec<CheckResult>,
    pub estimates: ResourceEstimates,
}

impl ResourceReport {
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

pub fn estimate_resources(flat: &HashMap<String, f64>, raw: &HashMap<String, serde_yaml::Value>, n_examples: usize) -> ResourceReport {
    let mut report = ResourceReport::default();

    let precision = raw.get("precision").or_else(|| raw.get("dtype"))
        .and_then(|v| v.as_str()).unwrap_or("bf16");
    let bpp = bytes_per_param(precision);
    let grad_bpp = bytes_per_param(precision);
    let gb = (1024u64 * 1024 * 1024) as f64;

    // Total params
    let mut total_params: u64 = 0;
    if let Some(p) = raw.get("total_params").or_else(|| raw.get("params")).and_then(|v| v.as_str()) {
        total_params = parse_param_count(p);
    }
    if total_params == 0 {
        let model_name = raw.get("model").or_else(|| raw.get("base_model"))
            .and_then(|v| v.as_str()).unwrap_or("");
        total_params = estimate_params_from_name(model_name);
    }
    if total_params == 0 {
        report.add("Parameter count", Severity::Warn, "Could not determine parameter count. Specify 'total_params' or 'model' in config.", "");
        return report;
    }

    // Method
    let method = raw.get("method").or_else(|| raw.get("training_method"))
        .and_then(|v| v.as_str()).unwrap_or("full").to_lowercase();
    let lora_r = flat.get("lora_r").or_else(|| flat.get("lora_rank")).copied();
    let is_lora = method.contains("lora") || lora_r.is_some();
    let is_qlora = method.contains("qlora") || raw.get("load_in_4bit").and_then(|v| v.as_bool()).unwrap_or(false);

    let trainable_params = if is_lora {
        let r = lora_r.unwrap_or(16.0) as u64;
        let frac = (r as f64 / 1600.0).min(0.05);
        (total_params as f64 * frac) as u64
    } else {
        total_params
    };

    // VRAM
    let (model_memory_gb, optimizer_memory_gb, gradient_memory_gb) = if is_lora {
        let base_bpp = if is_qlora { 0.5 } else { bpp };
        (
            total_params as f64 * base_bpp / gb,
            trainable_params as f64 * ADAMW_STATE_BYTES_PER_PARAM / gb,
            trainable_params as f64 * grad_bpp / gb,
        )
    } else {
        (
            total_params as f64 * bpp / gb,
            total_params as f64 * ADAMW_STATE_BYTES_PER_PARAM / gb,
            total_params as f64 * grad_bpp / gb,
        )
    };

    let batch = flat.get("batch_size").or_else(|| flat.get("per_device_train_batch_size")).copied().unwrap_or(1.0) as usize;
    let seq_len = flat.get("max_seq_len").or_else(|| flat.get("max_length")).copied().unwrap_or(512.0) as usize;
    let hidden = estimate_hidden_dim(total_params);
    let n_layers = estimate_n_layers(total_params);
    let activation_memory_gb = (batch * seq_len * hidden * n_layers * 2) as f64 / gb;

    let total_vram_gb = model_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb;

    // Steps
    let accum = flat.get("grad_accum_steps").or_else(|| flat.get("gradient_accumulation_steps")).copied().unwrap_or(1.0) as usize;
    let epochs = flat.get("epochs").or_else(|| flat.get("num_train_epochs")).copied().unwrap_or(1.0) as usize;
    let eff = batch * accum;
    let steps_per_epoch = if n_examples > 0 && eff > 0 { (n_examples + eff - 1) / eff } else { 0 };
    let total_steps = steps_per_epoch * epochs;

    let ms_per_step = estimate_ms_per_step(total_params, batch, seq_len);
    let estimated_time_minutes = total_steps as f64 * ms_per_step / 60_000.0;

    let cost_per_hour = flat.get("cost_per_hour").copied().unwrap_or(0.0);
    let estimated_cost_usd = if cost_per_hour > 0.0 && estimated_time_minutes > 0.0 {
        (estimated_time_minutes / 60.0) * cost_per_hour
    } else {
        0.0
    };

    // Store estimates
    report.estimates = ResourceEstimates {
        total_params,
        trainable_params,
        model_memory_gb,
        optimizer_memory_gb,
        gradient_memory_gb,
        activation_memory_gb,
        total_vram_gb,
        steps_per_epoch,
        total_steps,
        estimated_time_minutes,
        estimated_cost_usd,
    };

    // Now add all checks (no borrow conflict)
    report.add(
        "VRAM estimate",
        Severity::Info,
        &format!(
            "~{:.1}GB total (model: {:.1}, optimizer: {:.1}, grad: {:.1}, activations: {:.1})",
            total_vram_gb, model_memory_gb, optimizer_memory_gb, gradient_memory_gb, activation_memory_gb,
        ),
        "",
    );

    // GPU fit check
    if let Some(gpu_vram) = get_gpu_vram_bytes() {
        let gpu_gb = gpu_vram as f64 / gb;
        let util = total_vram_gb / gpu_gb;
        if util > 0.95 {
            report.add("VRAM fit", Severity::Warn, &format!("Estimated {:.1}GB exceeds 95% of {:.1}GB GPU. High OOM risk.", total_vram_gb, gpu_gb), "Reduce batch size, use gradient accumulation, or switch to LoRA/QLoRA.");
        } else if util > 0.85 {
            report.add("VRAM fit", Severity::Warn, &format!("Estimated {:.1}GB is {:.0}% of {:.1}GB GPU. Tight fit.", total_vram_gb, util * 100.0, gpu_gb), "");
        } else {
            report.add("VRAM fit", Severity::Info, &format!("Estimated {:.1}GB fits in {:.1}GB GPU ({:.0}% utilization)", total_vram_gb, gpu_gb, util * 100.0), "OK");
        }
    } else {
        report.add("VRAM fit", Severity::Info, "No GPU detected -- cannot verify fit", "");
    }

    // Steps
    if n_examples > 0 {
        report.add("Training steps", Severity::Info, &format!("{} steps/epoch x {} epochs = {} total (batch={}, accum={}, effective={})", steps_per_epoch, epochs, total_steps, batch, accum, eff), "");

        let hours = estimated_time_minutes / 60.0;
        if hours >= 1.0 {
            report.add("Time estimate", Severity::Info, &format!("~{hours:.1} hours (rough estimate)"), "");
        } else {
            report.add("Time estimate", Severity::Info, &format!("~{:.0} minutes (rough estimate)", estimated_time_minutes), "");
        }
    }

    // Cost
    if cost_per_hour > 0.0 && estimated_time_minutes > 0.0 {
        report.add("Cost estimate", Severity::Info, &format!("~${:.2} at ${cost_per_hour}/hr", estimated_cost_usd), "");
    } else {
        report.add("Cost estimate", Severity::Info, "$0 (local)", "");
    }

    // Param summary
    if trainable_params < total_params {
        let pct = trainable_params as f64 / total_params as f64 * 100.0;
        report.add("Parameters", Severity::Info, &format!("{}M total, {:.1}M trainable ({pct:.1}%)", total_params / 1_000_000, trainable_params as f64 / 1_000_000.0), "");
    } else {
        report.add("Parameters", Severity::Info, &format!("{}M parameters (all trainable)", total_params / 1_000_000), "");
    }

    report
}

fn estimate_params_from_name(name: &str) -> u64 {
    if let Some((_, spec)) = find_model(name) {
        return parse_param_count(spec.params);
    }
    let upper = name.to_uppercase();
    for (suffix, multiplier) in [("B", 1_000_000_000u64), ("M", 1_000_000)] {
        if let Some(pos) = upper.find(suffix) {
            if pos > 0 {
                let start = upper[..pos].rfind(|c: char| !c.is_ascii_digit() && c != '.').map(|i| i + 1).unwrap_or(0);
                if let Ok(n) = upper[start..pos].parse::<f64>() {
                    return (n * multiplier as f64) as u64;
                }
            }
        }
    }
    0
}

fn estimate_hidden_dim(total_params: u64) -> usize {
    if total_params > 5_000_000_000 { 4096 }
    else if total_params > 1_000_000_000 { 2048 }
    else if total_params > 100_000_000 { 768 }
    else { 512 }
}

fn estimate_n_layers(total_params: u64) -> usize {
    if total_params > 5_000_000_000 { 32 }
    else if total_params > 1_000_000_000 { 24 }
    else if total_params > 100_000_000 { 12 }
    else { 6 }
}

fn estimate_ms_per_step(total_params: u64, batch: usize, seq_len: usize) -> f64 {
    let base_ms = 100.0;
    let param_factor = total_params as f64 / 125_000_000.0;
    let batch_factor = batch as f64;
    let seq_factor = seq_len as f64 / 512.0;
    base_ms * param_factor * batch_factor * seq_factor
}
