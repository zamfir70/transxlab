/// Environment checks via Python probe subprocess.
use crate::knowledge::rules::Severity;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub status: Severity,
    pub message: String,
    pub detail: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EnvironmentReport {
    pub checks: Vec<CheckResult>,
}

impl EnvironmentReport {
    pub fn worst_severity(&self) -> Severity {
        if self.checks.iter().any(|c| c.status == Severity::Fail) {
            Severity::Fail
        } else if self.checks.iter().any(|c| c.status == Severity::Warn) {
            Severity::Warn
        } else {
            Severity::Info
        }
    }

    pub fn add(&mut self, name: &str, status: Severity, message: &str, detail: &str) {
        self.checks.push(CheckResult {
            name: name.to_string(),
            status,
            message: message.to_string(),
            detail: detail.to_string(),
        });
    }
}

/// GPU info returned by the Python probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_bytes: u64,
    pub vram_gb: f64,
    pub cuda_version: String,
    pub torch_version: String,
    pub bf16_supported: bool,
    pub cudnn_available: bool,
    pub compute_capability: String,
}

/// Result of running the probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub python_version: String,
    pub torch_available: bool,
    pub cuda_available: bool,
    pub gpus: Vec<GpuInfo>,
    pub error: Option<String>,
}

/// Run the Python probe to get environment info.
pub fn run_probe() -> ProbeResult {
    let probe_script = include_str!("../../probe.py");

    let result = Command::new("python")
        .args(["-c", probe_script])
        .output();

    match result {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                match serde_json::from_str::<ProbeResult>(&stdout) {
                    Ok(probe) => probe,
                    Err(e) => ProbeResult {
                        python_version: String::new(),
                        torch_available: false,
                        cuda_available: false,
                        gpus: vec![],
                        error: Some(format!("Probe parse error: {e}")),
                    },
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                ProbeResult {
                    python_version: String::new(),
                    torch_available: false,
                    cuda_available: false,
                    gpus: vec![],
                    error: Some(format!("Probe failed: {stderr}")),
                }
            }
        }
        Err(e) => ProbeResult {
            python_version: String::new(),
            torch_available: false,
            cuda_available: false,
            gpus: vec![],
            error: Some(format!("Could not run python: {e}")),
        },
    }
}

/// Build environment report from probe results.
pub fn check_environment(required_packages: &[&str]) -> EnvironmentReport {
    let mut report = EnvironmentReport::default();
    let probe = run_probe();

    if let Some(err) = &probe.error {
        report.add("Python probe", Severity::Warn, &format!("Probe failed: {err}"), "");
        // Still add what we can
        if !probe.python_version.is_empty() {
            report.add("Python version", Severity::Info, &format!("Python {}", probe.python_version), "OK");
        }
        return report;
    }

    // Python version
    if !probe.python_version.is_empty() {
        report.add(
            "Python version",
            Severity::Info,
            &format!("Python {}", probe.python_version),
            "OK",
        );
    }

    // PyTorch
    if probe.torch_available {
        if let Some(gpu) = probe.gpus.first() {
            report.add(
                "PyTorch installed",
                Severity::Info,
                &format!("torch {}", gpu.torch_version),
                "OK",
            );
        } else {
            report.add("PyTorch installed", Severity::Info, "torch available", "OK");
        }
    } else {
        report.add("PyTorch installed", Severity::Fail, "PyTorch not found", "pip install torch");
    }

    // CUDA
    if probe.cuda_available {
        report.add("CUDA available", Severity::Info, "CUDA available", "OK");
        if let Some(gpu) = probe.gpus.first() {
            report.add(
                "CUDA version",
                Severity::Info,
                &format!("CUDA {}", gpu.cuda_version),
                "",
            );
        }
        for (i, gpu) in probe.gpus.iter().enumerate() {
            report.add(
                &format!("GPU {i}"),
                Severity::Info,
                &format!("{} ({:.1}GB)", gpu.name, gpu.vram_gb),
                &format!("Compute capability: {}", gpu.compute_capability),
            );
        }
        if let Some(gpu) = probe.gpus.first() {
            if gpu.cudnn_available {
                report.add("cuDNN", Severity::Info, "cuDNN available", "");
            } else {
                report.add("cuDNN", Severity::Warn, "cuDNN not available", "Training may be slower");
            }
            if gpu.bf16_supported {
                report.add("bf16 support", Severity::Info, "bf16 supported", "");
            } else {
                report.add("bf16 support", Severity::Info, "bf16 not supported (will use fp16)", "");
            }
        }
    } else {
        report.add(
            "CUDA available",
            Severity::Warn,
            "No CUDA -- CPU training only",
            "Training will be significantly slower without GPU",
        );
    }

    // Required packages (checked via probe in future, for now just note them)
    for pkg in required_packages {
        report.add(
            &format!("Package: {pkg}"),
            Severity::Info,
            &format!("{pkg} (check manually)"),
            "",
        );
    }

    report
}

/// Get GPU VRAM in bytes from probe, or None.
pub fn get_gpu_vram_bytes() -> Option<u64> {
    let probe = run_probe();
    probe.gpus.first().map(|g| g.vram_bytes)
}
