/// TransXLab CLI: The training architect.
/// Manual arg parsing (no clap), following TransXform patterns.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process;

use transxlab::data_strategy::analyzer::{self, QualityReport};
use transxlab::data_strategy::sources;
use transxlab::data_strategy::strategy;
use transxlab::design::architecture::{self, ArchitectureSpec};
use transxlab::design::configgen;
use transxlab::design::cost;
use transxlab::design::interview::{self, DesignInputs};
use transxlab::design::spec;
use transxlab::knowledge::hub;
use transxlab::knowledge::models;
use transxlab::knowledge::rules::Severity;
use transxlab::preflight::runner::{run_preflight, PreflightOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "setup" | "preflight" => cmd_setup(&args[2..]),
        "design" => cmd_design(&args[2..]),
        "data" => cmd_data(&args[2..]),
        "full" => cmd_full(&args[2..]),
        "diagnose" => cmd_diagnose(&args[2..]),
        "--help" | "-h" | "help" => {
            print_usage();
            process::exit(0);
        }
        "--version" | "-V" => {
            eprintln!("transxlab 0.1.0");
            process::exit(0);
        }
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("TransXLab: The training architect.");
    eprintln!("Design, validate, and preflight ML training runs.");
    eprintln!();
    eprintln!("Usage: transxlab <command> [options]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  full      Run all levels: preflight + design + data strategy");
    eprintln!("  setup     Level 1: Preflight validation (environment, data, config, resources)");
    eprintln!("  design    Level 2: Architecture recommendation from design inputs");
    eprintln!("  data      Level 3: Data quality analysis and strategy recommendation");
    eprintln!("  diagnose  Postmortem: analyze training logs for failure patterns");
    eprintln!();
    eprintln!("Global options:");
    eprintln!("  --fail-on <warn|fail>  Exit non-zero on warnings (warn) or only failures (fail)");
    eprintln!("  --json                 Emit machine-readable JSON summary to stdout");
    eprintln!();
    eprintln!("Run 'transxlab <command> --help' for command-specific options.");
}

/// Severity threshold for exit codes.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FailOn {
    Fail, // exit non-zero only on hard failures
    Warn, // exit non-zero on any warning (default for setup)
}

fn exit_code_for(worst: Severity, fail_on: FailOn) -> i32 {
    match (worst, fail_on) {
        (Severity::Fail, _) => 2,
        (Severity::Warn, FailOn::Warn) => 1,
        (Severity::Warn, FailOn::Fail) => 0,
        (Severity::Info, _) => 0,
    }
}

// ---------------------------------------------------------------------------
// setup (preflight)
// ---------------------------------------------------------------------------

fn cmd_setup(args: &[String]) {
    let mut config_path: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("./transxlab_output");
    let mut dry_run = false;
    let mut fix = false;
    let mut json_output = false;
    let mut verbose = false;
    let mut quiet = false;
    let mut fail_on = FailOn::Warn;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                i += 1;
                config_path = Some(PathBuf::from(&args[i]));
            }
            "--data-dir" | "-d" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            "--output-dir" | "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--fail-on" => {
                i += 1;
                fail_on = parse_fail_on(&args[i]);
            }
            "--dry-run" => dry_run = true,
            "--fix" => fix = true,
            "--json" => json_output = true,
            "--verbose" | "-v" => verbose = true,
            "--quiet" | "-q" => quiet = true,
            "--help" | "-h" => {
                eprintln!("transxlab setup: Preflight validation");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -c, --config <path>      Training config (YAML/JSON)");
                eprintln!("  -d, --data-dir <path>    Training data directory");
                eprintln!("  -o, --output-dir <path>  Report output directory [default: ./transxlab_output]");
                eprintln!("      --fail-on <level>    Exit non-zero threshold: warn (default) or fail");
                eprintln!("      --dry-run            Show what checks would run");
                eprintln!("      --fix                Auto-fix simple issues");
                eprintln!("      --json               Emit JSON report");
                eprintln!("  -v, --verbose            Verbose output");
                eprintln!("  -q, --quiet              Minimal output");
                process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    let opts = PreflightOptions {
        config_path: config_path.as_deref(),
        data_dir: data_dir.as_deref(),
        output_dir: &output_dir,
        dry_run,
        fix,
        json_output,
        verbose,
        quiet,
    };

    let exit_code = run_preflight(&opts);
    let worst = match exit_code {
        0 => Severity::Info,
        1 => Severity::Warn,
        _ => Severity::Fail,
    };
    process::exit(exit_code_for(worst, fail_on));
}

// ---------------------------------------------------------------------------
// design
// ---------------------------------------------------------------------------

fn cmd_design(args: &[String]) {
    let mut answers_file: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("./transxlab_output");
    let mut model_id: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--answers" | "-a" => {
                i += 1;
                answers_file = Some(PathBuf::from(&args[i]));
            }
            "--output-dir" | "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--model" | "-m" => {
                i += 1;
                model_id = Some(args[i].clone());
            }
            "--help" | "-h" => {
                eprintln!("transxlab design: Architecture recommendation");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -a, --answers <path>     YAML file with design inputs (non-interactive)");
                eprintln!("  -m, --model <id>         Model ID (HF repo like 'meta-llama/Llama-3-8B' or short name)");
                eprintln!("  -o, --output-dir <path>  Report output directory [default: ./transxlab_output]");
                process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    if let Some(ref mid) = model_id {
        resolve_and_print_model(mid);
    }

    let inputs = load_design_inputs(answers_file.as_deref());
    let arch_spec = architecture::recommend(&inputs);
    spec::print_design_report(&inputs, &arch_spec);

    let _ = std::fs::create_dir_all(&output_dir);
    let _ = interview::save_interview(&inputs, &output_dir.join("design_inputs.yaml"));
    let spec_dict = spec::build_spec_dict(&inputs, &arch_spec);
    let _ = spec::save_spec(&spec_dict, &output_dir.join("design_spec.yaml"));
    let _ = spec::save_design_report(&output_dir.join("design_report.md"), &inputs, &arch_spec);

    // Config generation
    let _ = configgen::save_configs(&output_dir, &inputs, &arch_spec);
    eprintln!("Generated training configs: configs/hf_trainer.yaml, configs/axolotl.yaml, configs/llamafactory.yaml");

    // Cost estimation
    let avg_seq = (inputs.input_seq_len + inputs.output_seq_len) / 2;
    let epochs = arch_spec.schedule.as_ref().map(|s| s.epochs).unwrap_or(3);
    let cost_report = cost::estimate_costs(
        arch_spec.estimated_vram_gb,
        arch_spec.estimated_params,
        inputs.data_size.max(1000),
        epochs,
        avg_seq.max(256),
        &arch_spec.training_method,
    );
    eprintln!();
    eprintln!("{}", cost::format_cost_report(&cost_report));
    let _ = cost::save_cost_report(&output_dir.join("cost_estimate.md"), &cost_report);

    eprintln!();
    eprintln!("Design artifacts saved to {}/", output_dir.display());
    process::exit(0);
}

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------

fn cmd_data(args: &[String]) {
    let mut data_dir: Option<PathBuf> = None;
    let mut design_spec: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("./transxlab_output");
    let mut _input_field = "input".to_string();
    let mut output_field = "output".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" | "-d" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            "--design-spec" | "-s" => {
                i += 1;
                design_spec = Some(PathBuf::from(&args[i]));
            }
            "--output-dir" | "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--input-field" => {
                i += 1;
                _input_field = args[i].clone();
            }
            "--output-field" => {
                i += 1;
                output_field = args[i].clone();
            }
            "--help" | "-h" => {
                eprintln!("transxlab data: Data quality analysis and strategy");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -d, --data-dir <path>      Training data directory");
                eprintln!("  -s, --design-spec <path>   Design spec YAML from 'transxlab design'");
                eprintln!("  -o, --output-dir <path>    Report output directory [default: ./transxlab_output]");
                eprintln!("      --input-field <name>   Input field name [default: input]");
                eprintln!("      --output-field <name>  Output field name [default: output]");
                process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    let (task_type, task_domain) = load_task_from_spec(design_spec.as_deref());
    run_data_pipeline(data_dir.as_deref(), &output_dir, &output_field, &task_type, &task_domain);
    process::exit(0);
}

// ---------------------------------------------------------------------------
// full — the unified pipeline
// ---------------------------------------------------------------------------

fn cmd_full(args: &[String]) {
    let mut config_path: Option<PathBuf> = None;
    let mut answers_file: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("./transxlab_output");
    let mut fix = false;
    let mut json_output = false;
    let mut verbose = false;
    let mut quiet = false;
    let mut fail_on = FailOn::Warn;
    let mut output_field = "output".to_string();
    let mut model_id: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                i += 1;
                config_path = Some(PathBuf::from(&args[i]));
            }
            "--answers" | "-a" => {
                i += 1;
                answers_file = Some(PathBuf::from(&args[i]));
            }
            "--data-dir" | "-d" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            "--output-dir" | "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--model" | "-m" => {
                i += 1;
                model_id = Some(args[i].clone());
            }
            "--fail-on" => {
                i += 1;
                fail_on = parse_fail_on(&args[i]);
            }
            "--fix" => fix = true,
            "--json" => json_output = true,
            "--verbose" | "-v" => verbose = true,
            "--quiet" | "-q" => quiet = true,
            "--output-field" => {
                i += 1;
                output_field = args[i].clone();
            }
            "--help" | "-h" => {
                eprintln!("transxlab full: Run all levels (preflight + design + data strategy)");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -c, --config <path>       Training config (YAML/JSON)");
                eprintln!("  -a, --answers <path>      Design inputs YAML (required for design level)");
                eprintln!("  -d, --data-dir <path>     Training data directory");
                eprintln!("  -m, --model <id>          Model ID (HF repo or short name)");
                eprintln!("  -o, --output-dir <path>   Report output directory [default: ./transxlab_output]");
                eprintln!("      --fail-on <level>     Exit non-zero threshold: warn (default) or fail");
                eprintln!("      --fix                 Auto-fix simple issues");
                eprintln!("      --json                Emit JSON summary to stdout");
                eprintln!("  -v, --verbose             Verbose output");
                eprintln!("  -q, --quiet               Minimal output");
                eprintln!("      --output-field <name> Output field name [default: output]");
                process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    if let Some(ref mid) = model_id {
        resolve_and_print_model(mid);
    }

    let _ = std::fs::create_dir_all(&output_dir);
    let mut worst_severity = Severity::Info;
    let mut json_sections = serde_json::Map::new();

    // -----------------------------------------------------------------------
    // Level 1: Preflight
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("=== Level 1: Preflight ===");
    let preflight_code = run_preflight(&PreflightOptions {
        config_path: config_path.as_deref(),
        data_dir: data_dir.as_deref(),
        output_dir: &output_dir,
        dry_run: false,
        fix,
        json_output: true, // always save JSON for unified report
        verbose,
        quiet,
    });

    let preflight_sev = match preflight_code {
        0 => Severity::Info,
        1 => Severity::Warn,
        _ => Severity::Fail,
    };
    worst_severity = worse(worst_severity, preflight_sev);

    if json_output {
        json_sections.insert("preflight".into(), serde_json::json!({
            "exit_code": preflight_code,
            "severity": format!("{}", preflight_sev),
        }));
    }

    // -----------------------------------------------------------------------
    // Level 2: Design
    // -----------------------------------------------------------------------
    let design_result: Option<(DesignInputs, ArchitectureSpec)> = if answers_file.is_some() {
        eprintln!();
        eprintln!("=== Level 2: Design ===");
        let inputs = load_design_inputs(answers_file.as_deref());
        let arch_spec = architecture::recommend(&inputs);
        spec::print_design_report(&inputs, &arch_spec);

        let _ = interview::save_interview(&inputs, &output_dir.join("design_inputs.yaml"));
        let spec_dict = spec::build_spec_dict(&inputs, &arch_spec);
        let _ = spec::save_spec(&spec_dict, &output_dir.join("design_spec.yaml"));
        let _ = spec::save_design_report(
            &output_dir.join("design_report.md"),
            &inputs,
            &arch_spec,
        );

        // Config generation
        let _ = configgen::save_configs(&output_dir, &inputs, &arch_spec);

        // Cost estimation
        let avg_seq = (inputs.input_seq_len + inputs.output_seq_len) / 2;
        let n_epochs = arch_spec.schedule.as_ref().map(|s| s.epochs).unwrap_or(3);
        let cost_report = cost::estimate_costs(
            arch_spec.estimated_vram_gb,
            arch_spec.estimated_params,
            inputs.data_size.max(1000),
            n_epochs,
            avg_seq.max(256),
            &arch_spec.training_method,
        );
        eprintln!();
        eprintln!("{}", cost::format_cost_report(&cost_report));
        let _ = cost::save_cost_report(&output_dir.join("cost_estimate.md"), &cost_report);

        if json_output {
            let fitting: Vec<&cost::CostEstimate> = cost_report.estimates.iter().filter(|e| e.fits_vram).collect();
            let cheapest = fitting.first().map(|e| serde_json::json!({
                "gpu": e.gpu_name,
                "provider": e.provider,
                "estimated_cost": format!("${:.2}", e.estimated_cost),
                "estimated_hours": format!("{:.1}", e.estimated_hours),
            }));
            json_sections.insert("design".into(), serde_json::json!({
                "base_model": arch_spec.base_model,
                "architecture": arch_spec.architecture_type,
                "method": arch_spec.training_method,
                "estimated_vram_gb": arch_spec.estimated_vram_gb,
                "recommendations": arch_spec.recommendations.len(),
                "best_cost": cheapest,
            }));
        }

        Some((inputs, arch_spec))
    } else {
        eprintln!();
        eprintln!("=== Level 2: Design (skipped, no --answers) ===");
        None
    };

    // -----------------------------------------------------------------------
    // Level 3: Data Strategy
    // -----------------------------------------------------------------------
    let (task_type, task_domain) = if let Some((ref inputs, _)) = design_result {
        (inputs.task_type.as_str().to_string(), inputs.task_description.clone())
    } else {
        ("any".to_string(), String::new())
    };

    eprintln!();
    eprintln!("=== Level 3: Data Strategy ===");

    let quality_report = if let Some(dd) = data_dir.as_deref() {
        let examples = load_data_files(dd);
        if examples.is_empty() {
            eprintln!("No data files found to analyze.");
            None
        } else {
            eprintln!("Loaded {} examples from {}", examples.len(), dd.display());
            let report = analyzer::analyze_quality(&examples, &output_field, &task_type, 5000);
            print_quality_report(&report);
            Some(report)
        }
    } else {
        None
    };

    let data_sev = quality_report
        .as_ref()
        .map(|r| r.worst_severity())
        .unwrap_or(Severity::Info);
    worst_severity = worse(worst_severity, data_sev);

    let data_strategy = strategy::recommend_strategy(&task_type, "balanced");
    print_strategy(data_strategy);

    let srcs = sources::suggest_sources(&task_type, &task_domain);
    print_sources(&srcs);

    let _ = save_data_plan(
        &output_dir.join("data_plan.md"),
        quality_report.as_ref(),
        data_strategy,
        &srcs,
        &task_type,
    );

    if json_output {
        let mut data_json = serde_json::Map::new();
        data_json.insert("strategy".into(), data_strategy.name.into());
        data_json.insert("severity".into(), format!("{}", data_sev).into());
        if let Some(ref qr) = quality_report {
            if let Some(sb) = qr.metrics.self_bleu {
                data_json.insert("self_bleu".into(), sb.into());
            }
            data_json.insert("duplication_rate".into(), qr.metrics.duplication_rate.into());
            data_json.insert("n_analyzed".into(), (qr.metrics.n_analyzed as u64).into());
        }
        json_sections.insert("data_strategy".into(), data_json.into());
    }

    // -----------------------------------------------------------------------
    // Unified report
    // -----------------------------------------------------------------------
    let _ = save_unified_report(
        &output_dir.join("full_report.md"),
        preflight_code,
        design_result.as_ref(),
        quality_report.as_ref(),
        data_strategy,
        &task_type,
        worst_severity,
    );

    // JSON summary to stdout
    if json_output {
        json_sections.insert("verdict".into(), severity_name(worst_severity).into());
        json_sections.insert("exit_code".into(), exit_code_for(worst_severity, fail_on).into());
        json_sections.insert(
            "generated".into(),
            chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string().into(),
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::Value::Object(json_sections))
                .unwrap_or_default()
        );
    }

    eprintln!();
    eprintln!("=== Verdict: {} ===", severity_name(worst_severity).to_uppercase());
    if design_result.is_some() {
        eprintln!("TransXform spec: {}/design_spec.yaml", output_dir.display());
    }
    eprintln!("All reports saved to {}/", output_dir.display());

    process::exit(exit_code_for(worst_severity, fail_on));
}

// ---------------------------------------------------------------------------
// diagnose — postmortem mode
// ---------------------------------------------------------------------------

fn cmd_diagnose(args: &[String]) {
    let mut log_path: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("./transxlab_output");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--log" | "-l" => {
                i += 1;
                log_path = Some(PathBuf::from(&args[i]));
            }
            "--output-dir" | "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--help" | "-h" => {
                eprintln!("transxlab diagnose: Postmortem analysis of training logs");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -l, --log <path>         Training log file (JSON-lines or text)");
                eprintln!("  -o, --output-dir <path>  Report output directory [default: ./transxlab_output]");
                process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                process::exit(1);
            }
        }
        i += 1;
    }

    let log_file = match log_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --log <path> is required.");
            process::exit(1);
        }
    };

    let log_text = match std::fs::read_to_string(&log_file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to read log file: {}", e);
            process::exit(2);
        }
    };

    eprintln!("TransXLab Diagnose: Postmortem Analysis");
    eprintln!("Log: {}", log_file.display());
    eprintln!();

    let entries = parse_log_entries(&log_text);
    if entries.is_empty() {
        eprintln!("No parseable log entries found.");
        process::exit(1);
    }
    eprintln!("Parsed {} log entries.", entries.len());

    let diagnoses = run_diagnosis(&entries);

    let _ = std::fs::create_dir_all(&output_dir);

    if diagnoses.is_empty() {
        eprintln!();
        eprintln!("[OK] No known failure patterns detected.");
    } else {
        eprintln!();
        eprintln!("Detected {} potential issue(s):", diagnoses.len());
        eprintln!();
        for d in &diagnoses {
            eprintln!("  [!!] {} (confidence: {})", d.failure_mode, d.confidence);
            eprintln!("       Signal: {}", d.signal);
            eprintln!("       Consequence: {}", d.consequence);
            eprintln!("       Mitigation: {}", d.mitigation);
            if !d.source.is_empty() {
                eprintln!("       Source: {}", d.source);
            }
            eprintln!();
        }
    }

    let _ = save_diagnosis_report(
        &output_dir.join("diagnosis_report.md"),
        &log_file,
        entries.len(),
        &diagnoses,
    );
    eprintln!("Diagnosis report saved to {}/diagnosis_report.md", output_dir.display());

    let exit_code = if diagnoses.is_empty() { 0 } else { 1 };
    process::exit(exit_code);
}

// ---------------------------------------------------------------------------
// Log parsing & diagnosis engine
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct LogEntry {
    step: Option<u64>,
    loss: Option<f64>,
    grad_norm: Option<f64>,
    eval_loss: Option<f64>,
    metrics: HashMap<String, f64>,
}

#[derive(Debug)]
struct Diagnosis {
    failure_mode: String,
    confidence: String,
    signal: String,
    consequence: String,
    mitigation: String,
    source: String,
}

fn parse_log_entries(text: &str) -> Vec<LogEntry> {
    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(obj) = val.as_object() {
                let mut entry = LogEntry {
                    step: obj.get("step").or_else(|| obj.get("global_step")).and_then(|v| v.as_u64()),
                    loss: obj.get("loss").or_else(|| obj.get("train_loss")).and_then(|v| v.as_f64()),
                    grad_norm: obj.get("grad_norm").or_else(|| obj.get("gradient_norm")).and_then(|v| v.as_f64()),
                    eval_loss: obj.get("eval_loss").or_else(|| obj.get("val_loss")).and_then(|v| v.as_f64()),
                    metrics: HashMap::new(),
                };
                for (k, v) in obj {
                    if let Some(f) = v.as_f64() {
                        entry.metrics.insert(k.clone(), f);
                    }
                }
                entries.push(entry);
            }
        }
    }
    entries
}

fn run_diagnosis(entries: &[LogEntry]) -> Vec<Diagnosis> {
    let mut diagnoses = Vec::new();

    let losses: Vec<(u64, f64)> = entries.iter().filter_map(|e| Some((e.step?, e.loss?))).collect();
    let grad_norms: Vec<(u64, f64)> = entries.iter().filter_map(|e| Some((e.step?, e.grad_norm?))).collect();
    let eval_losses: Vec<(u64, f64)> = entries.iter().filter_map(|e| Some((e.step?, e.eval_loss?))).collect();

    // 1. NaN/Inf loss (check first — most critical)
    for entry in entries {
        if let Some(loss) = entry.loss {
            if loss.is_nan() || loss.is_infinite() {
                diagnoses.push(Diagnosis {
                    failure_mode: "NaN/Inf Loss".into(),
                    confidence: "high".into(),
                    signal: format!("Loss became {} at step {}.",
                        if loss.is_nan() { "NaN" } else { "Inf" },
                        entry.step.unwrap_or(0)),
                    consequence: "Training has catastrophically failed. All subsequent updates are garbage.".into(),
                    mitigation: "Reduce learning rate, check for bad data (empty/corrupt examples), add gradient clipping.".into(),
                    source: "empirical".into(),
                });
                break;
            }
        }
    }

    // 2. Loss divergence
    if losses.len() >= 10 {
        let n = losses.len().min(10);
        let last_avg = losses[losses.len()-n..].iter().map(|(_, l)| l).sum::<f64>() / n as f64;
        let first_avg = losses[..n].iter().map(|(_, l)| l).sum::<f64>() / n as f64;
        if last_avg > first_avg * 2.0 && last_avg > 5.0 {
            diagnoses.push(Diagnosis {
                failure_mode: "Loss Divergence".into(),
                confidence: "high".into(),
                signal: format!("Loss increased from {:.3} (early) to {:.3} (late).", first_avg, last_avg),
                consequence: "Model is not learning. Weights are being destroyed.".into(),
                mitigation: "Reduce learning rate by 10x. Check for data issues or gradient explosion.".into(),
                source: "AC-v2-postmortem".into(),
            });
        }
    }

    // 3. Loss plateau
    if losses.len() >= 20 {
        let mid = losses.len() / 2;
        let second_avg = losses[mid..].iter().map(|(_, l)| l).sum::<f64>() / (losses.len() - mid) as f64;
        let second_std = (losses[mid..].iter().map(|(_, l)| (l - second_avg).powi(2)).sum::<f64>()
            / (losses.len() - mid) as f64).sqrt();
        let first_avg = losses[..mid].iter().map(|(_, l)| l).sum::<f64>() / mid as f64;

        if second_std < 0.01 * second_avg.abs().max(0.01)
            && (second_avg - first_avg).abs() < 0.05 * first_avg.abs().max(0.01)
        {
            diagnoses.push(Diagnosis {
                failure_mode: "Loss Plateau".into(),
                confidence: "medium".into(),
                signal: format!("Loss plateaued at ~{:.4} for second half (std={:.6}).", second_avg, second_std),
                consequence: "Model stopped learning. Possibly stuck in local minimum or lr too low.".into(),
                mitigation: "Try lr warmup restart, briefly increase lr, or add more diverse data.".into(),
                source: "empirical".into(),
            });
        }
    }

    // 4. Gradient explosion
    if grad_norms.len() >= 5 {
        let max_norm = grad_norms.iter().map(|(_, g)| *g).fold(0.0f64, f64::max);
        let avg_norm = grad_norms.iter().map(|(_, g)| g).sum::<f64>() / grad_norms.len() as f64;
        if max_norm > 100.0 || (max_norm > avg_norm * 10.0 && avg_norm > 1.0) {
            diagnoses.push(Diagnosis {
                failure_mode: "Gradient Explosion".into(),
                confidence: "high".into(),
                signal: format!("Gradient norm spike: max={:.1}, avg={:.3}.", max_norm, avg_norm),
                consequence: "Training instability, potential NaN loss.".into(),
                mitigation: "Add gradient clipping (max_grad_norm=1.0), reduce learning rate.".into(),
                source: "literature".into(),
            });
        }
    }

    // 5. Overfitting
    if !eval_losses.is_empty() && losses.len() >= 10 {
        let last_train = losses.last().map(|(_, l)| *l).unwrap_or(0.0);
        let first_train = losses.first().map(|(_, l)| *l).unwrap_or(0.0);
        let last_eval = eval_losses.last().map(|(_, l)| *l).unwrap_or(0.0);
        let first_eval = eval_losses.first().map(|(_, l)| *l).unwrap_or(0.0);

        if last_train < first_train * 0.8 && last_eval > first_eval * 1.1 {
            diagnoses.push(Diagnosis {
                failure_mode: "Overfitting".into(),
                confidence: "high".into(),
                signal: format!("Train loss: {:.4}->{:.4}, eval loss: {:.4}->{:.4}.",
                    first_train, last_train, first_eval, last_eval),
                consequence: "Model memorizing training data instead of generalizing.".into(),
                mitigation: "Reduce epochs, add regularization (dropout, weight decay), or more data.".into(),
                source: "literature".into(),
            });
        }
    }

    // 6. Mode collapse (pw_cos)
    let pw_cos: Vec<f64> = entries.iter()
        .filter_map(|e| e.metrics.get("pw_cos").or_else(|| e.metrics.get("pairwise_cosine")).copied())
        .collect();
    if pw_cos.len() >= 3 {
        let last_3: Vec<f64> = pw_cos.iter().rev().take(3).copied().collect();
        if last_3.iter().all(|&v| v > 0.9) {
            diagnoses.push(Diagnosis {
                failure_mode: "Representation Collapse (Mode Collapse)".into(),
                confidence: "high".into(),
                signal: format!("Pairwise cosine > 0.9 for 3 consecutive evals (last: {:.3}).", last_3[0]),
                consequence: "All outputs collapsing to similar representations. Diversity gone.".into(),
                mitigation: "Add diversity loss (weight >= 0.3), reduce lr, check for template contamination.".into(),
                source: "AC-v2-postmortem".into(),
            });
        }
    }

    diagnoses
}

fn save_diagnosis_report(
    path: &Path, log_file: &Path, n_entries: usize, diagnoses: &[Diagnosis],
) -> std::io::Result<()> {
    let mut lines = vec![
        "# TransXLab Diagnosis Report".to_string(),
        String::new(),
        format!("Generated: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")),
        format!("Log file: {}", log_file.display()),
        format!("Entries parsed: {}", n_entries),
        String::new(),
    ];

    if diagnoses.is_empty() {
        lines.push("## Result: No Issues Detected".into());
        lines.push(String::new());
        lines.push("No known failure patterns matched in the training log.".into());
    } else {
        lines.push(format!("## Result: {} Issue(s) Detected", diagnoses.len()));
        lines.push(String::new());
        for (i, d) in diagnoses.iter().enumerate() {
            lines.push(format!("### {}. {} (confidence: {})", i + 1, d.failure_mode, d.confidence));
            lines.push(String::new());
            lines.push(format!("**Signal:** {}", d.signal));
            lines.push(format!("**Consequence:** {}", d.consequence));
            lines.push(format!("**Mitigation:** {}", d.mitigation));
            if !d.source.is_empty() {
                lines.push(format!("*Source: {}*", d.source));
            }
            lines.push(String::new());
        }
    }
    std::fs::write(path, lines.join("\n"))
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Resolve a model identifier: try HF Hub first (if it looks like a repo), then static KB.
fn resolve_and_print_model(model_id: &str) {
    eprintln!();
    eprintln!("=== Model Lookup: {} ===", model_id);

    // If it contains '/', treat as HF repo ID and try fetching
    if model_id.contains('/') {
        eprintln!("Fetching config from HuggingFace Hub...");
        match hub::fetch_hub_config(model_id) {
            Ok(info) => {
                let spec = info.to_model_spec();
                eprintln!("  Architecture: {}", info.architecture);
                eprintln!("  Params: {}", spec.params);
                eprintln!("  VRAM (bf16): {}", spec.vram_bf16);
                eprintln!("  VRAM (QLoRA): {}", spec.vram_qlora);
                eprintln!("  Max seq len: {}", spec.max_seq_len);
                if let Some(dt) = &info.torch_dtype {
                    eprintln!("  Default dtype: {}", dt);
                }
                if let Some(h) = info.hidden_size {
                    eprintln!("  Hidden size: {}", h);
                }
                if let Some(l) = info.num_layers {
                    eprintln!("  Layers: {}", l);
                }
                eprintln!();
                return;
            }
            Err(e) => {
                eprintln!("  Hub fetch failed: {}", e);
                eprintln!("  Falling back to static knowledge base...");
            }
        }
    }

    // Fallback: static knowledge base
    if let Some((key, spec)) = models::find_model(model_id) {
        eprintln!("  Found in knowledge base: {} ({})", spec.name, key);
        eprintln!("  Params: {}", spec.params);
        eprintln!("  VRAM (bf16): {}", spec.vram_bf16);
        eprintln!("  VRAM (QLoRA): {}", spec.vram_qlora);
        eprintln!("  Max seq len: {}", spec.max_seq_len);
        eprintln!("  Architecture: {}", spec.architecture);
        eprintln!("  Good for: {}", spec.good_for.join(", "));
    } else {
        eprintln!("  Model '{}' not found in knowledge base.", model_id);
        eprintln!("  Tip: Use a HF repo ID (e.g., 'meta-llama/Llama-3-8B') for auto-detection.");
    }
    eprintln!();
}

fn parse_fail_on(s: &str) -> FailOn {
    match s.to_lowercase().as_str() {
        "fail" => FailOn::Fail,
        _ => FailOn::Warn,
    }
}

fn load_design_inputs(answers_file: Option<&Path>) -> DesignInputs {
    if let Some(path) = answers_file {
        match interview::load_interview_from_file(path) {
            Ok(inp) => inp,
            Err(e) => {
                eprintln!("Failed to load answers file: {}", e);
                process::exit(2);
            }
        }
    } else {
        eprintln!("Error: --answers <path> is required (interactive mode not supported in Rust build).");
        process::exit(1);
    }
}

fn load_task_from_spec(spec_path: Option<&Path>) -> (String, String) {
    if let Some(path) = spec_path {
        if path.exists() {
            if let Ok(text) = std::fs::read_to_string(path) {
                if let Ok(val) = serde_yaml::from_str::<serde_yaml::Value>(&text) {
                    let tt = val.get("task").and_then(|t| t.get("type")).and_then(|v| v.as_str()).unwrap_or("any");
                    let td = val.get("task").and_then(|t| t.get("description")).and_then(|v| v.as_str()).unwrap_or("");
                    eprintln!("Loaded design spec: task type = {}", tt);
                    return (tt.to_string(), td.to_string());
                }
            }
        }
    }
    ("any".to_string(), String::new())
}

fn run_data_pipeline(
    data_dir: Option<&Path>, output_dir: &Path, output_field: &str, task_type: &str, task_domain: &str,
) {
    let quality_report = if let Some(dd) = data_dir {
        let examples = load_data_files(dd);
        if examples.is_empty() {
            eprintln!("No data files found to analyze.");
            None
        } else {
            eprintln!("Loaded {} examples from {}", examples.len(), dd.display());
            let report = analyzer::analyze_quality(&examples, output_field, task_type, 5000);
            print_quality_report(&report);
            Some(report)
        }
    } else {
        None
    };

    let strat = strategy::recommend_strategy(task_type, "balanced");
    print_strategy(strat);

    let srcs = sources::suggest_sources(task_type, task_domain);
    print_sources(&srcs);

    let _ = std::fs::create_dir_all(output_dir);
    let _ = save_data_plan(
        &output_dir.join("data_plan.md"), quality_report.as_ref(), strat, &srcs, task_type,
    );
    eprintln!("Data plan saved to {}/data_plan.md", output_dir.display());
}

fn load_data_files(dir: &Path) -> Vec<HashMap<String, String>> {
    let mut all: Vec<HashMap<String, String>> = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return all };
    let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    files.sort_by_key(|e| e.file_name());

    for entry in files {
        let path = entry.path();
        if !path.is_file() { continue; }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if !matches!(ext.as_str(), "jsonl" | "json") { continue; }

        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let parse_obj = |obj: &serde_json::Map<String, serde_json::Value>| -> HashMap<String, String> {
                    obj.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()
                };
                if ext == "jsonl" {
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        if let Ok(serde_json::Value::Object(map)) = serde_json::from_str(line) {
                            all.push(parse_obj(&map));
                        }
                    }
                } else if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(&text) {
                    for obj in arr {
                        if let serde_json::Value::Object(map) = obj {
                            all.push(parse_obj(&map));
                        }
                    }
                }
                eprintln!("Loaded data from {}", path.display());
            }
            Err(e) => eprintln!("Failed to load {}: {}", path.display(), e),
        }
    }
    all
}

fn print_quality_report(report: &QualityReport) {
    eprintln!();
    eprintln!("Data Quality Analysis");
    for check in &report.checks {
        let sym = match check.status { Severity::Info => "[OK]", Severity::Warn => "[!!]", Severity::Fail => "[XX]" };
        eprintln!("  {} {}: {}", sym, check.name, check.message);
    }
    eprintln!();
}

fn print_strategy(strat: &strategy::DataStrategy) {
    eprintln!("Recommended Data Strategy: {}", strat.name);
    eprintln!("  {}", strat.description);
    eprintln!("  Minimum examples: {}", strat.minimum_examples);
    eprintln!("  Recommended: {}", strat.recommended_examples);
    eprintln!("  Split ratio: {}", strat.split_ratio);
    eprintln!("  Format: {}", strat.format_notes);
    if !strat.quality_checks.is_empty() {
        eprintln!("  Quality checks:");
        for qc in strat.quality_checks { eprintln!("    - {}", qc); }
    }
    if !strat.anti_patterns.is_empty() {
        eprintln!("  Anti-patterns to avoid:");
        for ap in strat.anti_patterns { eprintln!("    - {}", ap); }
    }
    eprintln!();
}

fn print_sources(srcs: &[&sources::DataSource]) {
    if srcs.is_empty() { return; }
    eprintln!("Suggested Data Sources");
    for source in srcs.iter().take(5) {
        eprintln!("  {} ({})", source.name, source.domain);
        eprintln!("    {}", source.description);
        for w in source.warnings { eprintln!("    Warning: {}", w); }
    }
    eprintln!();
}

fn worse(a: Severity, b: Severity) -> Severity {
    match (a, b) {
        (Severity::Fail, _) | (_, Severity::Fail) => Severity::Fail,
        (Severity::Warn, _) | (_, Severity::Warn) => Severity::Warn,
        _ => Severity::Info,
    }
}

fn severity_name(s: Severity) -> &'static str {
    match s { Severity::Fail => "blocked", Severity::Warn => "warnings", Severity::Info => "ready" }
}

fn save_data_plan(
    path: &Path, quality_report: Option<&QualityReport>,
    strat: &strategy::DataStrategy, srcs: &[&sources::DataSource], task_type: &str,
) -> std::io::Result<()> {
    let mut lines = vec![
        "# TransXLab Data Plan".to_string(), String::new(),
        format!("Generated: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")),
        format!("Task type: {}", task_type), String::new(),
    ];

    if let Some(report) = quality_report {
        lines.push("## Data Quality Analysis".into());
        lines.push(String::new());
        for check in &report.checks {
            let sym = match check.status { Severity::Info => "OK", Severity::Warn => "WARN", Severity::Fail => "FAIL" };
            lines.push(format!("- {} **{}**: {}", sym, check.name, check.message));
            if !check.detail.is_empty() && check.detail != "OK" {
                lines.push(format!("  - {}", check.detail));
            }
        }
        lines.push(String::new());
        let m = &report.metrics;
        lines.push("### Metrics Summary".into());
        lines.push(String::new());
        if let Some(sb) = m.self_bleu { lines.push(format!("- Self-BLEU: {:.3}", sb)); }
        if let Some(ld) = m.lexical_diversity { lines.push(format!("- Lexical diversity (root TTR): {:.3}", ld)); }
        lines.push(format!("- Duplication rate: {:.1}%", m.duplication_rate * 100.0));
        lines.push(format!("- Mean output length: {:.0} words", m.length_mean));
        lines.push(format!("- Examples analyzed: {}", m.n_analyzed));
        lines.push(String::new());
    }

    lines.push("## Recommended Strategy".into());
    lines.push(String::new());
    lines.push(format!("### {}", strat.name));
    lines.push(String::new());
    lines.push(strat.description.to_string());
    lines.push(String::new());
    lines.push(format!("- Minimum examples: {}", strat.minimum_examples));
    lines.push(format!("- Recommended: {}", strat.recommended_examples));
    lines.push(format!("- Split ratio: {}", strat.split_ratio));
    lines.push(format!("- Format: {}", strat.format_notes));
    lines.push(String::new());
    if !strat.quality_checks.is_empty() {
        lines.push("**Quality checks:**".into());
        for qc in strat.quality_checks { lines.push(format!("- {}", qc)); }
        lines.push(String::new());
    }
    if !strat.anti_patterns.is_empty() {
        lines.push("**Anti-patterns to avoid:**".into());
        for ap in strat.anti_patterns { lines.push(format!("- {}", ap)); }
        lines.push(String::new());
    }
    lines.push("## Suggested Sources".into());
    lines.push(String::new());
    for source in srcs.iter().take(5) {
        lines.push(format!("### {}", source.name));
        lines.push(format!("- Domain: {}", source.domain));
        lines.push(format!("- {}", source.description));
        if !source.url.is_empty() { lines.push(format!("- Reference: {}", source.url)); }
        for w in source.warnings { lines.push(format!("- **Warning:** {}", w)); }
        lines.push(String::new());
    }
    std::fs::write(path, lines.join("\n"))
}

fn save_unified_report(
    path: &Path, preflight_code: i32,
    design: Option<&(DesignInputs, ArchitectureSpec)>,
    quality: Option<&QualityReport>,
    strat: &strategy::DataStrategy, task_type: &str, worst: Severity,
) -> std::io::Result<()> {
    let mut lines = vec![
        "# TransXLab Full Report".to_string(), String::new(),
        format!("Generated: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")),
        format!("Verdict: **{}**", severity_name(worst).to_uppercase()),
        String::new(),
        "## Level 1: Preflight".into(), String::new(),
        format!("Result: {} (see preflight_report.md)",
            match preflight_code { 0 => "READY", 1 => "WARNINGS", _ => "BLOCKED" }),
        String::new(),
    ];

    if let Some((inputs, arch_spec)) = design {
        lines.push("## Level 2: Design".into());
        lines.push(String::new());
        lines.push(format!("- Task: {}", inputs.task_description));
        lines.push(format!("- Approach: {}", inputs.approach));
        lines.push(format!("- Base model: {}", arch_spec.base_model));
        lines.push(format!("- Architecture: {}", arch_spec.architecture_type));
        lines.push(format!("- Method: {}", arch_spec.training_method));
        lines.push(format!("- Estimated VRAM: {:.1}GB", arch_spec.estimated_vram_gb));
        lines.push(format!("- Recommendations: {} (see design_report.md)", arch_spec.recommendations.len()));
        lines.push(String::new());
    }

    lines.push("## Level 3: Data Strategy".into());
    lines.push(String::new());
    lines.push(format!("- Task type: {}", task_type));
    lines.push(format!("- Strategy: {}", strat.name));
    if let Some(qr) = quality {
        if let Some(sb) = qr.metrics.self_bleu { lines.push(format!("- Self-BLEU: {:.3}", sb)); }
        lines.push(format!("- Duplication: {:.1}%", qr.metrics.duplication_rate * 100.0));
        lines.push(format!("- Examples analyzed: {}", qr.metrics.n_analyzed));
    }
    lines.push("- See data_plan.md for full details".into());
    lines.push(String::new());

    lines.push("## Artifacts".into());
    lines.push(String::new());
    lines.push("- `preflight_report.md` -- Environment, config, resource validation".into());
    lines.push("- `preflight_report.json` -- Machine-readable preflight results".into());
    if design.is_some() {
        lines.push("- `design_spec.yaml` -- Architecture spec (TransXform handoff)".into());
        lines.push("- `design_report.md` -- Design recommendations".into());
    }
    lines.push("- `data_plan.md` -- Data quality analysis and strategy".into());
    lines.push("- `full_report.md` -- This unified report".into());
    lines.push(String::new());

    std::fs::write(path, lines.join("\n"))
}
