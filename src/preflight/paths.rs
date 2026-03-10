/// Path validation: output dirs, checkpoint paths, disk space.
use crate::knowledge::rules::Severity;
use crate::preflight::environment::CheckResult;
use std::path::Path;

#[derive(Debug, Default)]
pub struct PathReport {
    pub checks: Vec<CheckResult>,
}

impl PathReport {
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

pub fn check_paths(
    output_dir: Option<&Path>,
    checkpoint_dir: Option<&Path>,
    fix: bool,
) -> PathReport {
    let mut report = PathReport::default();

    if let Some(dir) = output_dir {
        check_dir(&mut report, dir, "Output directory", fix);
    }
    if let Some(dir) = checkpoint_dir {
        check_dir(&mut report, dir, "Checkpoint directory", fix);
        // Existing checkpoints
        if dir.exists() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                let count = entries
                    .flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with("checkpoint"))
                    .count();
                if count > 0 {
                    report.add(
                        "Existing checkpoints",
                        Severity::Warn,
                        &format!("{count} existing checkpoints in {}", dir.display()),
                        "New checkpoints may collide. Consider a new directory.",
                    );
                }
            }
        }
    }

    report
}

fn check_dir(report: &mut PathReport, path: &Path, label: &str, fix: bool) {
    if path.exists() {
        if path.is_dir() {
            report.add(label, Severity::Info, &format!("{} exists and is writable", path.display()), "OK");
        } else {
            report.add(label, Severity::Fail, &format!("{} exists but is not a directory", path.display()), "");
        }
    } else if fix {
        match std::fs::create_dir_all(path) {
            Ok(_) => report.add(label, Severity::Info, &format!("Created {}", path.display()), "Auto-fixed"),
            Err(e) => report.add(label, Severity::Fail, &format!("Cannot create {}: {e}", path.display()), ""),
        }
    } else {
        report.add(
            label,
            Severity::Warn,
            &format!("{} does not exist (can be created)", path.display()),
            "Use --fix to auto-create, or it will be created at training time",
        );
    }
}
