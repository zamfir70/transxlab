/// Cloud cost estimation for GPU training runs.

#[derive(Debug, Clone)]
pub struct GpuTier {
    pub name: &'static str,
    pub vram_gb: f64,
    pub tflops_bf16: f64,
    pub prices: &'static [(&'static str, f64)], // (provider, $/hr)
}

static GPU_TIERS: &[GpuTier] = &[
    GpuTier {
        name: "RTX 3090",
        vram_gb: 24.0,
        tflops_bf16: 35.6,
        prices: &[("RunPod", 0.44), ("Vast.ai", 0.30)],
    },
    GpuTier {
        name: "RTX 4090",
        vram_gb: 24.0,
        tflops_bf16: 82.6,
        prices: &[("RunPod", 0.69), ("Vast.ai", 0.50)],
    },
    GpuTier {
        name: "A100 40GB",
        vram_gb: 40.0,
        tflops_bf16: 312.0,
        prices: &[("RunPod", 1.64), ("Lambda", 1.10), ("AWS", 4.10)],
    },
    GpuTier {
        name: "A100 80GB",
        vram_gb: 80.0,
        tflops_bf16: 312.0,
        prices: &[("RunPod", 2.49), ("Lambda", 1.29), ("AWS", 5.12)],
    },
    GpuTier {
        name: "H100 80GB",
        vram_gb: 80.0,
        tflops_bf16: 989.0,
        prices: &[("RunPod", 3.89), ("Lambda", 2.49), ("AWS", 8.20)],
    },
    GpuTier {
        name: "L40S",
        vram_gb: 48.0,
        tflops_bf16: 181.0,
        prices: &[("RunPod", 1.14), ("Lambda", 0.99)],
    },
    GpuTier {
        name: "A10G",
        vram_gb: 24.0,
        tflops_bf16: 31.2,
        prices: &[("AWS", 1.21)],
    },
];

#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub gpu_name: String,
    pub provider: String,
    pub hourly_rate: f64,
    pub estimated_hours: f64,
    pub estimated_cost: f64,
    pub fits_vram: bool,
}

#[derive(Debug, Clone)]
pub struct CostReport {
    pub estimates: Vec<CostEstimate>,
    pub training_hours_estimate: f64,
    pub vram_required_gb: f64,
    pub param_count: u64,
}

/// Estimate training time in GPU-hours.
///
/// Rough model: time = (6 * params * tokens) / (tflops * 1e12 * utilization)
/// For fine-tuning: tokens ≈ data_size * avg_seq_len * epochs
/// Utilization assumed ~30% for fine-tuning (overhead from data loading, eval, etc.)
fn estimate_training_hours(
    param_count: u64,
    data_size: usize,
    epochs: usize,
    avg_seq_len: usize,
    tflops: f64,
    method: &str,
) -> f64 {
    if param_count == 0 || tflops == 0.0 {
        return 0.0;
    }

    let tokens = data_size as f64 * avg_seq_len as f64 * epochs as f64;

    // FLOPs per token depends on method
    let flops_per_token = match method {
        "lora" | "qlora" => {
            // LoRA trains ~1-2% of params but forward pass uses all
            // Roughly: 2 * params (forward) + 2 * 0.02 * params (backward) ≈ 2.04 * params
            2.04 * param_count as f64
        }
        _ => {
            // Full training: ~6 * params per token (forward + backward + optimizer)
            6.0 * param_count as f64
        }
    };

    let total_flops = flops_per_token * tokens;
    let utilization = 0.30; // conservative for typical training
    let effective_tflops = tflops * 1e12 * utilization;

    if effective_tflops == 0.0 {
        return 0.0;
    }

    total_flops / effective_tflops / 3600.0 // convert seconds to hours
}

/// Estimate costs across GPU tiers and providers.
pub fn estimate_costs(
    vram_required_gb: f64,
    param_count: u64,
    data_size: usize,
    epochs: usize,
    avg_seq_len: usize,
    method: &str,
) -> CostReport {
    let mut estimates = Vec::new();

    for gpu in GPU_TIERS {
        let fits = gpu.vram_gb >= vram_required_gb;
        let hours = estimate_training_hours(
            param_count,
            data_size,
            epochs,
            avg_seq_len,
            gpu.tflops_bf16,
            method,
        );

        for &(provider, hourly) in gpu.prices {
            estimates.push(CostEstimate {
                gpu_name: gpu.name.to_string(),
                provider: provider.to_string(),
                hourly_rate: hourly,
                estimated_hours: hours,
                estimated_cost: hours * hourly,
                fits_vram: fits,
            });
        }
    }

    // Sort: fitting GPUs first, then by cost
    estimates.sort_by(|a, b| {
        b.fits_vram
            .cmp(&a.fits_vram)
            .then(a.estimated_cost.partial_cmp(&b.estimated_cost).unwrap_or(std::cmp::Ordering::Equal))
    });

    let ref_hours = estimates
        .iter()
        .find(|e| e.fits_vram)
        .map(|e| e.estimated_hours)
        .unwrap_or(0.0);

    CostReport {
        estimates,
        training_hours_estimate: ref_hours,
        vram_required_gb,
        param_count,
    }
}

/// Format cost report for display.
pub fn format_cost_report(report: &CostReport) -> String {
    let mut lines = Vec::new();
    lines.push("Cloud Cost Estimates".to_string());
    lines.push(format!(
        "  Model: ~{:.0}M params | VRAM needed: ~{:.1}GB",
        report.param_count as f64 / 1e6,
        report.vram_required_gb,
    ));
    lines.push(String::new());

    lines.push(format!(
        "  {:<14} {:<10} {:>8} {:>10} {:>12}  {}",
        "GPU", "Provider", "$/hr", "Hours", "Total Cost", ""
    ));
    lines.push(format!("  {}", "-".repeat(70)));

    let fitting: Vec<&CostEstimate> = report.estimates.iter().filter(|e| e.fits_vram).collect();
    let not_fitting: Vec<&CostEstimate> = report.estimates.iter().filter(|e| !e.fits_vram).collect();

    for est in &fitting {
        lines.push(format!(
            "  {:<14} {:<10} {:>7.2} {:>9.1}h {:>11.2}",
            est.gpu_name, est.provider, est.hourly_rate, est.estimated_hours, est.estimated_cost,
        ));
    }

    if !not_fitting.is_empty() && !fitting.is_empty() {
        lines.push(String::new());
        lines.push("  -- Below GPUs do NOT have enough VRAM --".to_string());
        for est in not_fitting.iter().take(3) {
            lines.push(format!(
                "  {:<14} {:<10} {:>7.2} {:>9.1}h {:>11.2}  (insufficient VRAM)",
                est.gpu_name, est.provider, est.hourly_rate, est.estimated_hours, est.estimated_cost,
            ));
        }
    }

    if fitting.is_empty() {
        lines.push("  No single GPU has enough VRAM. Consider multi-GPU or model parallelism.".to_string());
    } else {
        let cheapest = fitting.first().unwrap();
        lines.push(String::new());
        lines.push(format!(
            "  Best value: {} on {} — ~${:.2} ({:.1}h @ ${:.2}/hr)",
            cheapest.gpu_name, cheapest.provider, cheapest.estimated_cost,
            cheapest.estimated_hours, cheapest.hourly_rate,
        ));
    }

    lines.join("\n")
}

/// Save cost report as markdown.
pub fn save_cost_report(
    path: &std::path::Path,
    report: &CostReport,
) -> std::io::Result<()> {
    let mut lines = Vec::new();
    lines.push("# TransXLab Cost Estimate".to_string());
    lines.push(String::new());
    lines.push(format!(
        "Generated: {}",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));
    lines.push(format!(
        "Parameters: ~{:.0}M",
        report.param_count as f64 / 1e6
    ));
    lines.push(format!(
        "VRAM required: ~{:.1}GB",
        report.vram_required_gb
    ));
    lines.push(String::new());

    lines.push("## GPU Options".to_string());
    lines.push(String::new());
    lines.push("| GPU | Provider | $/hr | Est. Hours | Est. Cost | Fits? |".to_string());
    lines.push("|-----|----------|------|------------|-----------|-------|".to_string());

    for est in &report.estimates {
        let fits = if est.fits_vram { "Yes" } else { "No" };
        lines.push(format!(
            "| {} | {} | ${:.2} | {:.1}h | ${:.2} | {} |",
            est.gpu_name, est.provider, est.hourly_rate, est.estimated_hours,
            est.estimated_cost, fits,
        ));
    }

    lines.push(String::new());

    let fitting: Vec<&CostEstimate> = report.estimates.iter().filter(|e| e.fits_vram).collect();
    if let Some(best) = fitting.first() {
        lines.push(format!(
            "**Best value:** {} on {} — **${:.2}** ({:.1}h @ ${:.2}/hr)",
            best.gpu_name, best.provider, best.estimated_cost,
            best.estimated_hours, best.hourly_rate,
        ));
    }
    lines.push(String::new());
    lines.push("*Note: Prices are approximate community/on-demand rates and may vary. Training time estimates assume ~30% GPU utilization (typical for fine-tuning with data loading, evaluation, and checkpointing overhead).*".to_string());

    std::fs::write(path, lines.join("\n"))
}
