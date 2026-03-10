/// Dimension and schedule heuristics for architecture recommendations.
use crate::knowledge::rules::{ScratchDimensions, SCRATCH_HEURISTICS};

#[derive(Debug, Clone)]
pub struct TrainingSchedule {
    pub lr: f64,
    pub warmup_steps: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub grad_accum_steps: usize,
    pub effective_batch: usize,
    pub precision: String,
    pub weight_decay: f64,
    pub label_smoothing: f64,
    pub dropout: f64,
}

#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub r: u32,
    pub alpha: u32,
    pub target_modules: Vec<String>,
    pub dropout: f64,
}

pub fn select_scratch_dimensions(param_budget_str: &str) -> &'static ScratchDimensions {
    let budget = parse_param_budget(param_budget_str);

    let mut best = &SCRATCH_HEURISTICS[0];
    let mut best_diff = (budget as i64 - approx_params(best) as i64).unsigned_abs();

    for dims in SCRATCH_HEURISTICS.iter() {
        let diff = (budget as i64 - approx_params(dims) as i64).unsigned_abs();
        if diff < best_diff {
            best = dims;
            best_diff = diff;
        }
    }

    best
}

pub fn recommend_training_schedule(
    approach: &str,
    method: &str,
    data_size: usize,
    param_count: u64,
    vram_gb: f64,
) -> TrainingSchedule {
    let precision = "bf16".to_string();

    let base_batch = estimate_fitting_batch(param_count, vram_gb, method);
    let target_effective = target_effective_batch(data_size);
    let grad_accum = std::cmp::max(1, target_effective / base_batch);
    let effective_batch = base_batch * grad_accum;

    if approach == "scratch" {
        let lr = if param_count < 200_000_000 { 1e-3 } else { 3e-4 };
        let epochs = scratch_epochs(data_size);
        let steps_per_epoch = if effective_batch > 0 {
            data_size.div_ceil(effective_batch)
        } else {
            1
        };
        let warmup = std::cmp::max(
            100,
            ((steps_per_epoch * epochs) as f64 * 0.05).ceil() as usize,
        );
        TrainingSchedule {
            lr,
            warmup_steps: warmup,
            epochs,
            batch_size: base_batch,
            grad_accum_steps: grad_accum,
            effective_batch,
            precision,
            weight_decay: 0.01,
            label_smoothing: 0.1,
            dropout: 0.1,
        }
    } else {
        let lr = if method == "lora" || method == "qlora" {
            2e-4
        } else {
            3e-5
        };
        let epochs = finetune_epochs(data_size);
        let steps_per_epoch = std::cmp::max(
            1,
            data_size.div_ceil(effective_batch),
        );
        let total_steps = steps_per_epoch * epochs;
        let warmup = std::cmp::min((total_steps as f64 * 0.06) as usize, 500);
        let warmup = std::cmp::max(warmup, 10);

        let dropout = if method == "lora" || method == "qlora" {
            0.05
        } else {
            0.0
        };

        TrainingSchedule {
            lr,
            warmup_steps: warmup,
            epochs,
            batch_size: base_batch,
            grad_accum_steps: grad_accum,
            effective_batch,
            precision,
            weight_decay: 0.01,
            label_smoothing: 0.0,
            dropout,
        }
    }
}

pub fn recommend_lora_config(
    task_type: &str,
    creativity: &str,
    base_model: &str,
) -> LoRAConfig {
    let r = if creativity == "creativity" || task_type == "generate" {
        32
    } else if task_type == "classify" {
        8
    } else {
        16
    };

    let alpha = r * 2;

    let base_lower = base_model.to_lowercase();
    let target_modules = if ["llama", "mistral", "qwen"]
        .iter()
        .any(|k| base_lower.contains(k))
    {
        vec![
            "q_proj".into(),
            "v_proj".into(),
            "k_proj".into(),
            "o_proj".into(),
        ]
    } else if ["t5", "flan"].iter().any(|k| base_lower.contains(k)) {
        vec!["q".into(), "v".into(), "k".into(), "o".into()]
    } else {
        vec!["q_proj".into(), "v_proj".into()]
    };

    LoRAConfig {
        r,
        alpha,
        target_modules,
        dropout: 0.05,
    }
}

pub fn parse_param_budget(s: &str) -> u64 {
    let s = s.trim().to_uppercase();
    if let Some(rest) = s.strip_suffix('B') {
        (rest.parse::<f64>().unwrap_or(0.0) * 1e9) as u64
    } else if let Some(rest) = s.strip_suffix('M') {
        (rest.parse::<f64>().unwrap_or(0.0) * 1e6) as u64
    } else if let Some(rest) = s.strip_suffix('K') {
        (rest.parse::<f64>().unwrap_or(0.0) * 1e3) as u64
    } else {
        s.parse::<u64>().unwrap_or(0)
    }
}

fn approx_params(dims: &ScratchDimensions) -> u64 {
    let s = dims.approx_params.trim_start_matches('~');
    parse_param_budget(s)
}

fn estimate_fitting_batch(param_count: u64, vram_gb: f64, method: &str) -> usize {
    if param_count == 0 {
        return 4;
    }

    let gb = (1024u64 * 1024 * 1024) as f64;
    let (base_cost_gb, train_cost_gb) = if method == "lora" || method == "qlora" {
        let base = (param_count as f64 * 2.0) / gb;
        let train = (param_count as f64 * 0.01 * 12.0) / gb;
        (base, train)
    } else {
        let base = (param_count as f64 * 12.0) / gb;
        (base, 0.0)
    };

    let overhead_gb = base_cost_gb + train_cost_gb;
    let available = (vram_gb - overhead_gb - 1.0).max(1.0);
    let per_element_gb = (overhead_gb * 0.02).max(0.1);
    let batch = (available / per_element_gb).max(1.0) as usize;
    batch.min(32)
}

fn target_effective_batch(data_size: usize) -> usize {
    if data_size < 1000 {
        8
    } else if data_size < 10000 {
        16
    } else if data_size < 100000 {
        32
    } else {
        64
    }
}

fn scratch_epochs(data_size: usize) -> usize {
    if data_size < 10000 {
        20
    } else if data_size < 50000 {
        10
    } else if data_size < 200000 {
        5
    } else {
        3
    }
}

fn finetune_epochs(data_size: usize) -> usize {
    if data_size < 1000 {
        5
    } else if data_size < 5000 {
        3
    } else if data_size < 50000 {
        2
    } else {
        1
    }
}
