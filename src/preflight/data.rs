/// Data validation: file existence, parsing, schema, splits, overlap.
use crate::knowledge::rules::Severity;
use crate::preflight::environment::CheckResult;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Default)]
pub struct DataReport {
    pub checks: Vec<CheckResult>,
    pub train_count: usize,
    pub val_count: usize,
}

impl DataReport {
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

/// Load a JSONL file as a vec of serde_json::Value maps.
fn load_jsonl(path: &Path) -> Result<Vec<serde_json::Map<String, serde_json::Value>>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let val: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("Line {}: {e}", i + 1))?;
        if let serde_json::Value::Object(map) = val {
            records.push(map);
        } else {
            return Err(format!("Line {}: expected object", i + 1));
        }
    }
    Ok(records)
}

/// Load a JSON array file.
fn load_json(path: &Path) -> Result<Vec<serde_json::Map<String, serde_json::Value>>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let val: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
    match val {
        serde_json::Value::Array(arr) => {
            let mut records = Vec::new();
            for item in arr {
                if let serde_json::Value::Object(map) = item {
                    records.push(map);
                }
            }
            Ok(records)
        }
        _ => Err("Expected a JSON array".to_string()),
    }
}

/// Load a data file, auto-detecting format.
fn load_file(path: &Path) -> Result<Vec<serde_json::Map<String, serde_json::Value>>, String> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("jsonl") => load_jsonl(path),
        Some("json") => load_json(path),
        _ => load_jsonl(path), // Default to JSONL
    }
}

/// Find train/val files in a directory.
fn find_split_files(dir: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let mut train = None;
    let mut val = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if name.contains("train") {
                train = Some(entry.path());
            } else if name.contains("val") {
                val = Some(entry.path());
            }
        }
    }
    (train, val)
}

/// Simple MD5-like hash for overlap detection (using a basic hash).
fn hash_string(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Run data validation checks.
pub fn check_data(
    data_dir: Option<&Path>,
    train_path: Option<&Path>,
    val_path: Option<&Path>,
    input_field: &str,
    output_field: &str,
) -> DataReport {
    let mut report = DataReport::default();

    // Resolve file paths
    let (auto_train, auto_val) = if let Some(dir) = data_dir {
        if train_path.is_none() {
            find_split_files(dir)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let train = train_path.map(PathBuf::from).or(auto_train);
    let val = val_path.map(PathBuf::from).or(auto_val);

    let train = match train {
        Some(t) => t,
        None => {
            report.add("Training data", Severity::Fail, "No training data file found", "");
            return report;
        }
    };

    if !train.exists() {
        report.add(
            "Training data",
            Severity::Fail,
            &format!("File not found: {}", train.display()),
            "",
        );
        return report;
    }
    report.add(
        "Training data",
        Severity::Info,
        &format!("Found: {}", train.display()),
        "OK",
    );

    // Parse
    let train_data = match load_file(&train) {
        Ok(d) => d,
        Err(e) => {
            report.add("Parse training data", Severity::Fail, &format!("Failed to parse: {e}"), "");
            return report;
        }
    };

    report.train_count = train_data.len();
    report.add(
        "Training examples",
        Severity::Info,
        &format!("{} examples", train_data.len()),
        "",
    );

    if train_data.is_empty() {
        report.add("Training data", Severity::Fail, "Training file is empty", "");
        return report;
    }

    // Required fields
    let first = &train_data[0];
    for (field, label) in [(input_field, "input"), (output_field, "output")] {
        if first.contains_key(field) {
            report.add(&format!("Field: {field}"), Severity::Info, &format!("{label} field present"), "OK");
        } else {
            let available: Vec<_> = first.keys().cloned().collect();
            report.add(
                &format!("Field: {field}"),
                Severity::Fail,
                &format!("{label} field '{field}' not found. Available: {}", available.join(", ")),
                "",
            );
        }
    }

    // Empty examples
    let empty_input = train_data
        .iter()
        .filter(|ex| {
            ex.get(input_field)
                .map(|v| v.as_str().unwrap_or("").trim().is_empty())
                .unwrap_or(true)
        })
        .count();
    let empty_output = train_data
        .iter()
        .filter(|ex| {
            ex.get(output_field)
                .map(|v| v.as_str().unwrap_or("").trim().is_empty())
                .unwrap_or(true)
        })
        .count();

    if empty_input > 0 {
        report.add(
            "Empty inputs",
            Severity::Warn,
            &format!("{empty_input} examples have empty {input_field}"),
            &format!("{:.1}% of training data", empty_input as f64 / train_data.len() as f64 * 100.0),
        );
    }
    if empty_output > 0 {
        report.add(
            "Empty outputs",
            Severity::Warn,
            &format!("{empty_output} examples have empty {output_field}"),
            &format!("{:.1}% of training data", empty_output as f64 / train_data.len() as f64 * 100.0),
        );
    }

    // Schema consistency
    let expected_keys: HashSet<_> = first.keys().cloned().collect();
    let inconsistent = train_data
        .iter()
        .filter(|ex| {
            let keys: HashSet<_> = ex.keys().cloned().collect();
            keys != expected_keys
        })
        .count();

    if inconsistent > 0 {
        report.add(
            "Schema consistency",
            Severity::Warn,
            &format!("{inconsistent} examples have different fields than the first example"),
            "",
        );
    } else {
        report.add("Schema consistency", Severity::Info, "All examples have consistent schema", "OK");
    }

    // Validation split
    let val_data = if let Some(vp) = &val {
        if vp.exists() {
            match load_file(vp) {
                Ok(d) => {
                    report.val_count = d.len();
                    report.add("Validation data", Severity::Info, &format!("{} examples", d.len()), "OK");
                    Some(d)
                }
                Err(e) => {
                    report.add("Validation data", Severity::Warn, &format!("Failed to parse val: {e}"), "");
                    None
                }
            }
        } else {
            report.add("Validation data", Severity::Warn, &format!("Val file not found: {}", vp.display()), "");
            None
        }
    } else {
        report.add(
            "Validation split",
            Severity::Warn,
            "No validation split found",
            "Consider creating a train/val split to monitor overfitting",
        );
        None
    };

    // Train/val overlap
    if let Some(val_data) = &val_data {
        if first.contains_key(input_field) {
            let train_hashes: HashSet<u64> = train_data
                .iter()
                .map(|ex| hash_string(ex.get(input_field).and_then(|v| v.as_str()).unwrap_or("")))
                .collect();
            let val_hashes: HashSet<u64> = val_data
                .iter()
                .map(|ex| hash_string(ex.get(input_field).and_then(|v| v.as_str()).unwrap_or("")))
                .collect();
            let overlap = train_hashes.intersection(&val_hashes).count();
            if overlap > 0 {
                report.add(
                    "Train/val overlap",
                    Severity::Warn,
                    &format!("{overlap} overlapping examples between train and val"),
                    "This will inflate validation metrics and give false confidence",
                );
            } else {
                report.add("Train/val overlap", Severity::Info, "No overlap detected", "OK");
            }
        }
    }

    report
}
