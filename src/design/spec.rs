/// Design spec generation: emit YAML spec and markdown report.
use std::path::Path;

use crate::design::architecture::ArchitectureSpec;
use crate::design::interview::{DesignInputs, TrainingApproach};
use crate::error::Error;

/// Build the design_spec.yaml content as a serde_json::Value (then convert to YAML).
pub fn build_spec_dict(inputs: &DesignInputs, spec: &ArchitectureSpec) -> serde_json::Value {
    let mut d = serde_json::Map::new();

    // Task
    let mut task = serde_json::Map::new();
    task.insert("description".into(), inputs.task_description.clone().into());
    task.insert("type".into(), inputs.task_type.as_str().into());
    task.insert("input_format".into(), inputs.input_format.as_str().into());
    task.insert("output_format".into(), inputs.output_format.as_str().into());
    d.insert("task".into(), task.into());

    d.insert("approach".into(), inputs.approach.as_str().into());

    // Architecture
    let mut arch = serde_json::Map::new();
    arch.insert("type".into(), spec.architecture_type.clone().into());

    if inputs.approach == TrainingApproach::FineTune {
        arch.insert("base_model".into(), spec.base_model.clone().into());
        arch.insert("method".into(), spec.training_method.clone().into());
        if let Some(ref lora) = spec.lora_config {
            let mut lora_map = serde_json::Map::new();
            lora_map.insert("r".into(), (lora.r as u64).into());
            lora_map.insert("alpha".into(), (lora.alpha as u64).into());
            lora_map.insert(
                "target_modules".into(),
                lora.target_modules
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect::<Vec<_>>()
                    .into(),
            );
            lora_map.insert("dropout".into(), lora.dropout.into());
            arch.insert("lora".into(), lora_map.into());
        }
    } else if let Some(dims) = spec.scratch_dims {
        let mut dim_map = serde_json::Map::new();
        dim_map.insert("d_model".into(), (dims.d_model as u64).into());
        dim_map.insert("n_heads".into(), (dims.n_heads as u64).into());
        dim_map.insert(
            "n_encoder_layers".into(),
            (dims.n_encoder_layers as u64).into(),
        );
        dim_map.insert(
            "n_decoder_layers".into(),
            (dims.n_decoder_layers as u64).into(),
        );
        dim_map.insert("d_ff".into(), (dims.d_ff as u64).into());
        arch.insert("dimensions".into(), dim_map.into());
        arch.insert("vocab_size".into(), 32000u64.into());
        arch.insert("normalization".into(), "pre_norm".into());
        arch.insert("activation".into(), "gelu".into());
    }
    d.insert("architecture".into(), arch.into());

    // Training
    if let Some(ref sched) = spec.schedule {
        let mut training = serde_json::Map::new();
        training.insert("lr".into(), sched.lr.into());
        training.insert("warmup_steps".into(), (sched.warmup_steps as u64).into());
        training.insert("epochs".into(), (sched.epochs as u64).into());
        training.insert("batch_size".into(), (sched.batch_size as u64).into());
        training.insert(
            "grad_accum_steps".into(),
            (sched.grad_accum_steps as u64).into(),
        );
        training.insert(
            "effective_batch".into(),
            (sched.effective_batch as u64).into(),
        );
        training.insert("precision".into(), sched.precision.clone().into());
        training.insert("weight_decay".into(), sched.weight_decay.into());
        training.insert("dropout".into(), sched.dropout.into());
        if sched.label_smoothing > 0.0 {
            training.insert("label_smoothing".into(), sched.label_smoothing.into());
        }
        d.insert("training".into(), training.into());
    }

    // Loss
    let mut loss = serde_json::Map::new();
    loss.insert("primary".into(), spec.loss_function.clone().into());
    if !spec.extra_losses.is_empty() {
        loss.insert(
            "extra".into(),
            spec.extra_losses
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect::<Vec<_>>()
                .into(),
        );
    }
    d.insert("loss".into(), loss.into());

    // Estimates
    let mut estimates = serde_json::Map::new();
    estimates.insert("parameters".into(), spec.estimated_params.into());
    estimates.insert(
        "vram_training_gb".into(),
        ((spec.estimated_vram_gb * 10.0).round() / 10.0).into(),
    );
    d.insert("estimates".into(), estimates.into());

    // Data
    let mut data = serde_json::Map::new();
    data.insert(
        "expected_format".into(),
        inputs.input_format.as_str().into(),
    );
    data.insert(
        "minimum_examples".into(),
        (minimum_examples(inputs) as u64).into(),
    );
    data.insert(
        "split_ratio".into(),
        (if inputs.data_size < 5000 {
            "90/10"
        } else {
            "90/5/5"
        })
        .into(),
    );
    d.insert("data".into(), data.into());

    // TransXform handoff spec
    d.insert("transxform".into(), build_transxform_spec(inputs));

    serde_json::Value::Object(d)
}

pub fn save_spec(spec_dict: &serde_json::Value, path: &Path) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let yaml_str = serde_yaml::to_string(spec_dict).map_err(Error::Yaml)?;
    std::fs::write(path, yaml_str).map_err(Error::Io)?;
    Ok(())
}

pub fn save_design_report(
    path: &Path,
    inputs: &DesignInputs,
    spec: &ArchitectureSpec,
) -> Result<(), Error> {
    let mut lines = vec![
        "# TransXLab Design Report".to_string(),
        String::new(),
        format!(
            "Generated: {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ),
        String::new(),
        "## Task".to_string(),
        format!("- **Description:** {}", inputs.task_description),
        format!("- **Type:** {}", inputs.task_type),
        format!("- **Input:** {}", inputs.input_format),
        format!("- **Output:** {}", inputs.output_format),
        format!("- **Approach:** {}", inputs.approach),
        String::new(),
        "## Recommendations".to_string(),
        String::new(),
    ];

    for rec in &spec.recommendations {
        lines.push(format!("### {}", rec.key));
        lines.push(format!(
            "**{}** (confidence: {})",
            rec.value, rec.confidence.level
        ));
        lines.push(String::new());
        lines.push(rec.rationale.clone());
        if !rec.confidence.reason.is_empty() {
            lines.push(format!("*{}*", rec.confidence.reason));
        }
        lines.push(String::new());
    }

    lines.push("## Estimates".to_string());
    lines.push(format!(
        "- Parameters: {:.0}M",
        spec.estimated_params as f64 / 1e6
    ));
    lines.push(format!(
        "- VRAM (training): ~{:.1}GB",
        spec.estimated_vram_gb
    ));
    lines.push(String::new());

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    std::fs::write(path, lines.join("\n")).map_err(Error::Io)?;
    Ok(())
}

pub fn print_design_report(inputs: &DesignInputs, spec: &ArchitectureSpec) {
    eprintln!();
    eprintln!("TransXLab Design Report");
    eprintln!(
        "{}",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    );
    eprintln!();
    eprintln!(
        "Task: {}",
        if inputs.task_description.is_empty() {
            inputs.task_type.as_str()
        } else {
            &inputs.task_description
        }
    );
    eprintln!("Approach: {}", inputs.approach);
    eprintln!();

    eprintln!(
        "{:<20} {:<40} {:<10} {}",
        "Decision", "Recommendation", "Conf", "Rationale"
    );
    eprintln!("{}", "-".repeat(100));
    for rec in &spec.recommendations {
        let val = if rec.value.len() > 38 {
            format!("{}...", &rec.value[..35])
        } else {
            rec.value.clone()
        };
        eprintln!(
            "{:<20} {:<40} {:<10} {}",
            rec.key, val, rec.confidence.level, rec.rationale,
        );
    }
    eprintln!();
}

fn minimum_examples(inputs: &DesignInputs) -> usize {
    if inputs.approach == TrainingApproach::Scratch {
        10000
    } else if inputs.task_type.as_str() == "classify" {
        500
    } else {
        1000
    }
}

fn build_transxform_spec(inputs: &DesignInputs) -> serde_json::Value {
    let mut tf = serde_json::Map::new();

    let mut invariants: Vec<serde_json::Value> =
        vec!["grad_norm < 10.0".into()];

    let mut checkpoints = serde_json::Map::new();
    checkpoints.insert("save_every".into(), 500u64.into());
    checkpoints.insert("keep_best".into(), 3u64.into());
    tf.insert("checkpoints".into(), checkpoints.into());

    let task = inputs.task_type.as_str();
    match task {
        "generate" => {
            invariants.push("loss < 15.0 after step 100".into());
            tf.insert(
                "alerts".into(),
                vec![serde_json::Value::String(
                    "pw_cos > 0.9 for 3 consecutive evals -> representation collapse".into(),
                )]
                .into(),
            );

            let mut eval = serde_json::Map::new();
            eval.insert("novel_query_eval_every".into(), 500u64.into());
            eval.insert(
                "metrics".into(),
                vec!["loss", "generation_quality", "diversity"]
                    .into_iter()
                    .map(|s| serde_json::Value::String(s.into()))
                    .collect::<Vec<_>>()
                    .into(),
            );
            tf.insert("eval".into(), eval.into());

            let mut early_stop = serde_json::Map::new();
            early_stop.insert("metric".into(), "novel_query_generation_quality".into());
            early_stop.insert("patience".into(), 3u64.into());
            tf.insert("early_stop".into(), early_stop.into());
        }
        "classify" => {
            invariants.push("loss < 10.0 after step 50".into());

            let mut eval = serde_json::Map::new();
            eval.insert("eval_every".into(), 200u64.into());
            eval.insert(
                "metrics".into(),
                vec!["loss", "accuracy", "f1"]
                    .into_iter()
                    .map(|s| serde_json::Value::String(s.into()))
                    .collect::<Vec<_>>()
                    .into(),
            );
            tf.insert("eval".into(), eval.into());

            let mut early_stop = serde_json::Map::new();
            early_stop.insert("metric".into(), "val_accuracy".into());
            early_stop.insert("patience".into(), 5u64.into());
            tf.insert("early_stop".into(), early_stop.into());
        }
        _ => {
            let mut eval = serde_json::Map::new();
            eval.insert("eval_every".into(), 500u64.into());
            eval.insert(
                "metrics".into(),
                vec![serde_json::Value::String("loss".into())].into(),
            );
            tf.insert("eval".into(), eval.into());
        }
    }

    tf.insert("invariants".into(), invariants.into());

    serde_json::Value::Object(tf)
}
