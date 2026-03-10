/// Preflight runner: orchestrate all checks.
use crate::preflight::config::{check_config, load_config, ConfigReport};
use crate::preflight::data::{check_data, DataReport};
use crate::preflight::environment::check_environment;
use crate::preflight::paths::{check_paths, PathReport};
use crate::preflight::report::{print_report, save_json_report, save_markdown_report};
use crate::preflight::resources::{estimate_resources, ResourceReport};
use std::path::Path;

pub struct PreflightOptions<'a> {
    pub config_path: Option<&'a Path>,
    pub data_dir: Option<&'a Path>,
    pub output_dir: &'a Path,
    pub dry_run: bool,
    pub fix: bool,
    pub json_output: bool,
    pub verbose: bool,
    pub quiet: bool,
}

pub fn run_preflight(opts: &PreflightOptions) -> i32 {
    if opts.dry_run {
        print_dry_run(opts);
        return 0;
    }

    // Load config
    let (raw, flat) = if let Some(cp) = opts.config_path {
        match load_config(cp) {
            Ok((r, f)) => {
                if opts.verbose {
                    eprintln!("Loaded config from {}", cp.display());
                }
                (r, f)
            }
            Err(e) => {
                eprintln!("Failed to load config: {e}");
                return 2;
            }
        }
    } else {
        (Default::default(), Default::default())
    };

    // Infer required packages
    let required: Vec<&str> = {
        let mut pkgs = Vec::new();
        let method = raw.get("method").and_then(|v| v.as_str()).unwrap_or("");
        if method.contains("lora") || flat.contains_key("lora_r") {
            pkgs.push("peft");
        }
        if raw.contains_key("model") || raw.contains_key("base_model") {
            pkgs.push("transformers");
        }
        pkgs
    };

    // 1. Environment
    let env_report = check_environment(&required);

    // 2. Data
    let data_report: Option<DataReport> = {
        let effective_dir = opts.data_dir
            .or_else(|| raw.get("data_dir").and_then(|v| v.as_str()).map(Path::new));
        let train_file = raw.get("train_file").and_then(|v| v.as_str());
        let val_file = raw.get("val_file").and_then(|v| v.as_str());
        let input_field = raw.get("input_field").and_then(|v| v.as_str()).unwrap_or("input");
        let output_field = raw.get("output_field").and_then(|v| v.as_str()).unwrap_or("output");

        if effective_dir.is_some() || train_file.is_some() {
            Some(check_data(
                if train_file.is_none() { effective_dir } else { None },
                train_file.map(Path::new),
                val_file.map(Path::new),
                input_field,
                output_field,
            ))
        } else {
            None
        }
    };

    // 3. Config
    let config_report: Option<ConfigReport> = if !raw.is_empty() {
        let n = data_report.as_ref().map(|d| d.train_count).unwrap_or(0);
        Some(check_config(&raw, &flat, n))
    } else {
        None
    };

    // 4. Paths
    let path_report: Option<PathReport> = {
        let out_dir = raw.get("output_dir").and_then(|v| v.as_str());
        let ckpt_dir = raw.get("checkpoint_dir").or_else(|| raw.get("save_dir")).and_then(|v| v.as_str());
        if out_dir.is_some() || ckpt_dir.is_some() {
            Some(check_paths(
                out_dir.map(Path::new),
                ckpt_dir.map(Path::new),
                opts.fix,
            ))
        } else {
            None
        }
    };

    // 5. Resources
    let resource_report: Option<ResourceReport> = if !raw.is_empty() {
        let n = data_report.as_ref().map(|d| d.train_count).unwrap_or(0);
        Some(estimate_resources(&flat, &raw, n))
    } else {
        None
    };

    // Print
    let exit_code = print_report(
        &env_report,
        data_report.as_ref(),
        config_report.as_ref(),
        path_report.as_ref(),
        resource_report.as_ref(),
        opts.quiet,
    );

    // Save
    let _ = std::fs::create_dir_all(opts.output_dir);
    let _ = save_markdown_report(
        &opts.output_dir.join("preflight_report.md"),
        &env_report,
        data_report.as_ref(),
        config_report.as_ref(),
        path_report.as_ref(),
        resource_report.as_ref(),
    );

    if opts.json_output {
        let _ = save_json_report(
            &opts.output_dir.join("preflight_report.json"),
            &env_report,
            data_report.as_ref(),
            config_report.as_ref(),
            path_report.as_ref(),
            resource_report.as_ref(),
        );
    }

    eprintln!("Reports saved to {}/", opts.output_dir.display());
    exit_code
}

fn print_dry_run(opts: &PreflightOptions) {
    println!("TransXLab Preflight (dry run)");
    println!();
    println!("Would run the following checks:");
    println!("  1. Environment: Python, PyTorch, CUDA, GPU, required packages");
    if let Some(d) = opts.data_dir {
        println!("  2. Data: validate files in {}", d.display());
    } else {
        println!("  2. Data: (skipped, no --data-dir)");
    }
    if let Some(c) = opts.config_path {
        println!("  3. Config: validate hyperparameters in {}", c.display());
    } else {
        println!("  3. Config: (skipped, no --config)");
    }
    println!("  4. Paths: output dir, checkpoint dir writability");
    println!("  5. Resources: VRAM estimation, training time, cost");
}
