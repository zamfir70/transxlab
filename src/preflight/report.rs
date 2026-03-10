/// Report generation: console, markdown, JSON output.
use crate::knowledge::rules::Severity;
use crate::preflight::config::ConfigReport;
use crate::preflight::data::DataReport;
use crate::preflight::environment::{CheckResult, EnvironmentReport};
use crate::preflight::paths::PathReport;
use crate::preflight::resources::ResourceReport;
use chrono::Local;
use std::path::Path;

fn symbol(s: Severity) -> &'static str {
    match s {
        Severity::Info => "[OK]",
        Severity::Warn => "[!!]",
        Severity::Fail => "[XX]",
    }
}

fn verdict(severities: &[Severity]) -> (&'static str, Severity) {
    if severities.contains(&Severity::Fail) {
        ("BLOCKED", Severity::Fail)
    } else if severities.contains(&Severity::Warn) {
        ("WARNINGS", Severity::Warn)
    } else {
        ("READY", Severity::Info)
    }
}

fn collect_severities(
    env: &EnvironmentReport,
    data: Option<&DataReport>,
    config: Option<&ConfigReport>,
    paths: Option<&PathReport>,
    resources: Option<&ResourceReport>,
) -> Vec<Severity> {
    let mut s = vec![env.worst_severity()];
    if let Some(d) = data { s.push(d.worst_severity()); }
    if let Some(c) = config { s.push(c.worst_severity()); }
    if let Some(p) = paths { s.push(p.worst_severity()); }
    if let Some(r) = resources { s.push(r.worst_severity()); }
    s
}

/// Print report to stdout. Returns exit code.
pub fn print_report(
    env: &EnvironmentReport,
    data: Option<&DataReport>,
    config: Option<&ConfigReport>,
    paths: Option<&PathReport>,
    resources: Option<&ResourceReport>,
    quiet: bool,
) -> i32 {
    let sevs = collect_severities(env, data, config, paths, resources);
    let (verdict_text, verdict_sev) = verdict(&sevs);

    if quiet {
        println!("TransXLab Preflight: {verdict_text}");
        return match verdict_sev {
            Severity::Info => 0,
            Severity::Warn => 1,
            Severity::Fail => 2,
        };
    }

    println!();
    println!("TransXLab Preflight Report");
    println!("{}", Local::now().format("%Y-%m-%d %H:%M:%S"));
    println!();

    print_section("Environment", &env.checks);
    if let Some(d) = data { print_section("Data", &d.checks); }
    if let Some(c) = config { print_section("Config", &c.checks); }
    if let Some(p) = paths { print_section("Paths", &p.checks); }
    if let Some(r) = resources { print_section("Resources", &r.checks); }

    println!();
    println!("{} Verdict: {verdict_text}", symbol(verdict_sev));

    if verdict_sev == Severity::Fail {
        println!("Fix these issues before training:");
        let all_checks: Vec<&CheckResult> = env.checks.iter()
            .chain(data.map(|d| d.checks.iter()).into_iter().flatten())
            .chain(config.map(|c| c.checks.iter()).into_iter().flatten())
            .chain(paths.map(|p| p.checks.iter()).into_iter().flatten())
            .chain(resources.map(|r| r.checks.iter()).into_iter().flatten())
            .filter(|c| c.status == Severity::Fail)
            .collect();
        for c in all_checks {
            println!("  {} {}", symbol(Severity::Fail), c.message);
            if !c.detail.is_empty() && c.detail != "OK" {
                println!("    {}", c.detail);
            }
        }
    }

    println!();
    match verdict_sev {
        Severity::Info => 0,
        Severity::Warn => 1,
        Severity::Fail => 2,
    }
}

fn print_section(title: &str, checks: &[CheckResult]) {
    println!("  {title}");
    for c in checks {
        println!("    {} {:25} {}", symbol(c.status), c.name, c.message);
        if !c.detail.is_empty() && c.detail != "OK" {
            println!("    {:29} {}", "", c.detail);
        }
    }
    println!();
}

/// Save markdown report.
pub fn save_markdown_report(
    path: &Path,
    env: &EnvironmentReport,
    data: Option<&DataReport>,
    config: Option<&ConfigReport>,
    paths: Option<&PathReport>,
    resources: Option<&ResourceReport>,
) -> std::io::Result<()> {
    let mut lines = vec![
        "# TransXLab Preflight Report".to_string(),
        String::new(),
        format!("Generated: {}", Local::now().format("%Y-%m-%d %H:%M:%S")),
        String::new(),
    ];

    md_section(&mut lines, "Environment", &env.checks);
    if let Some(d) = data { md_section(&mut lines, "Data", &d.checks); }
    if let Some(c) = config { md_section(&mut lines, "Config", &c.checks); }
    if let Some(p) = paths { md_section(&mut lines, "Paths", &p.checks); }
    if let Some(r) = resources { md_section(&mut lines, "Resources", &r.checks); }

    let sevs = collect_severities(env, data, config, paths, resources);
    let (verdict_text, _) = verdict(&sevs);
    lines.push(format!("## Verdict: {verdict_text}"));
    lines.push(String::new());

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, lines.join("\n"))
}

fn md_section(lines: &mut Vec<String>, title: &str, checks: &[CheckResult]) {
    lines.push(format!("## {title}"));
    lines.push(String::new());
    for c in checks {
        let sym = match c.status {
            Severity::Info => "OK",
            Severity::Warn => "WARN",
            Severity::Fail => "FAIL",
        };
        lines.push(format!("- {sym} **{}**: {}", c.name, c.message));
        if !c.detail.is_empty() && c.detail != "OK" {
            lines.push(format!("  - {}", c.detail));
        }
    }
    lines.push(String::new());
}

/// Save JSON report.
pub fn save_json_report(
    path: &Path,
    env: &EnvironmentReport,
    data: Option<&DataReport>,
    config: Option<&ConfigReport>,
    paths: Option<&PathReport>,
    resources: Option<&ResourceReport>,
) -> std::io::Result<()> {
    let checks_to_json = |checks: &[CheckResult]| -> Vec<serde_json::Value> {
        checks.iter().map(|c| {
            serde_json::json!({
                "name": c.name,
                "status": c.status,
                "message": c.message,
                "detail": c.detail,
            })
        }).collect()
    };

    let sevs = collect_severities(env, data, config, paths, resources);
    let (verdict_text, verdict_sev) = verdict(&sevs);

    let mut report = serde_json::json!({
        "generated": Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        "environment": checks_to_json(&env.checks),
        "verdict": verdict_text,
        "exit_code": match verdict_sev { Severity::Info => 0, Severity::Warn => 1, Severity::Fail => 2 },
    });

    if let Some(d) = data {
        report["data"] = serde_json::json!(checks_to_json(&d.checks));
        report["data_counts"] = serde_json::json!({ "train": d.train_count, "val": d.val_count });
    }
    if let Some(c) = config { report["config"] = serde_json::json!(checks_to_json(&c.checks)); }
    if let Some(p) = paths { report["paths"] = serde_json::json!(checks_to_json(&p.checks)); }
    if let Some(r) = resources {
        report["resources"] = serde_json::json!(checks_to_json(&r.checks));
        report["estimates"] = serde_json::to_value(&r.estimates).unwrap_or_default();
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&report).unwrap_or_default())
}
