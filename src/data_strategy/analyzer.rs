/// Data quality analyzer: self-BLEU, lexical diversity, dedup, format compliance.
use std::collections::HashMap;

use crate::knowledge::rules::Severity;

#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub self_bleu: Option<f64>,
    pub lexical_diversity: Option<f64>,
    pub length_mean: f64,
    pub length_std: f64,
    pub duplication_rate: f64,
    pub empty_rate: f64,
    pub n_analyzed: usize,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            self_bleu: None,
            lexical_diversity: None,
            length_mean: 0.0,
            length_std: 0.0,
            duplication_rate: 0.0,
            empty_rate: 0.0,
            n_analyzed: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QualityCheck {
    pub name: String,
    pub status: Severity,
    pub message: String,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub checks: Vec<QualityCheck>,
    pub metrics: QualityMetrics,
}

impl QualityReport {
    fn new() -> Self {
        Self {
            checks: Vec::new(),
            metrics: QualityMetrics::default(),
        }
    }

    fn add(&mut self, name: &str, status: Severity, message: &str, detail: &str) {
        self.checks.push(QualityCheck {
            name: name.to_string(),
            status,
            message: message.to_string(),
            detail: detail.to_string(),
        });
    }

    pub fn worst_severity(&self) -> Severity {
        if self.checks.iter().any(|c| c.status == Severity::Fail) {
            Severity::Fail
        } else if self.checks.iter().any(|c| c.status == Severity::Warn) {
            Severity::Warn
        } else {
            Severity::Info
        }
    }
}

/// Analyze data quality beyond basic validation.
pub fn analyze_quality(
    examples: &[HashMap<String, String>],
    output_field: &str,
    task_type: &str,
    sample_size: usize,
) -> QualityReport {
    let mut report = QualityReport::new();

    if examples.is_empty() {
        report.add("Quality analysis", Severity::Warn, "No examples to analyze", "");
        return report;
    }

    // Sample if dataset is large
    let sample: Vec<&HashMap<String, String>> = if examples.len() > sample_size {
        report.add(
            "Sample size",
            Severity::Info,
            &format!("Analyzing {} of {} examples", sample_size, examples.len()),
            "",
        );
        // Simple deterministic sampling: take evenly spaced
        let step = examples.len() as f64 / sample_size as f64;
        (0..sample_size)
            .map(|i| &examples[(i as f64 * step) as usize])
            .collect()
    } else {
        report.add(
            "Sample size",
            Severity::Info,
            &format!("Analyzing all {} examples", examples.len()),
            "",
        );
        examples.iter().collect()
    };

    report.metrics.n_analyzed = sample.len();

    // Extract outputs
    let outputs: Vec<&str> = sample
        .iter()
        .map(|ex| ex.get(output_field).map(|s| s.as_str()).unwrap_or(""))
        .collect();
    let non_empty: Vec<&str> = outputs.iter().filter(|o| !o.trim().is_empty()).copied().collect();

    // Empty rate
    report.metrics.empty_rate = if !outputs.is_empty() {
        (outputs.len() - non_empty.len()) as f64 / outputs.len() as f64
    } else {
        0.0
    };
    if report.metrics.empty_rate > 0.05 {
        report.add(
            "Empty outputs",
            Severity::Warn,
            &format!("{:.1}% of outputs are empty", report.metrics.empty_rate * 100.0),
            "",
        );
    }

    if non_empty.is_empty() {
        report.add("Quality analysis", Severity::Warn, "All outputs are empty", "");
        return report;
    }

    // Length distribution
    let lengths: Vec<usize> = non_empty.iter().map(|o| o.split_whitespace().count()).collect();
    let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    report.metrics.length_mean = mean;
    let variance = lengths.iter().map(|l| (*l as f64 - mean).powi(2)).sum::<f64>() / lengths.len() as f64;
    report.metrics.length_std = variance.sqrt();
    report.add(
        "Output length",
        Severity::Info,
        &format!("Mean: {:.0} words, Std: {:.0} words", report.metrics.length_mean, report.metrics.length_std),
        "",
    );

    // Lexical diversity (root TTR)
    let mut all_tokens: Vec<String> = Vec::new();
    for o in &non_empty {
        for w in o.to_lowercase().split_whitespace() {
            all_tokens.push(w.to_string());
        }
    }
    if !all_tokens.is_empty() {
        let mut unique = std::collections::HashSet::new();
        for t in &all_tokens {
            unique.insert(t.as_str());
        }
        let total = all_tokens.len() as f64;
        report.metrics.lexical_diversity = Some(unique.len() as f64 / total.sqrt());
        report.add(
            "Lexical diversity",
            Severity::Info,
            &format!(
                "Root TTR: {:.3} ({} unique / {} total tokens)",
                report.metrics.lexical_diversity.unwrap(),
                unique.len(),
                all_tokens.len(),
            ),
            "",
        );
    }

    // Exact duplication (MD5)
    let hashes: Vec<String> = non_empty
        .iter()
        .map(|o| format!("{:x}", md5_hash(o)))
        .collect();
    let mut hash_counts: HashMap<&str, usize> = HashMap::new();
    for h in &hashes {
        *hash_counts.entry(h.as_str()).or_insert(0) += 1;
    }
    let n_duplicates: usize = hash_counts.values().filter(|&&c| c > 1).map(|c| c - 1).sum();
    report.metrics.duplication_rate = if !non_empty.is_empty() {
        n_duplicates as f64 / non_empty.len() as f64
    } else {
        0.0
    };
    if report.metrics.duplication_rate > 0.05 {
        report.add(
            "Duplication",
            Severity::Warn,
            &format!(
                "{:.1}% duplicate outputs ({} duplicates)",
                report.metrics.duplication_rate * 100.0,
                n_duplicates,
            ),
            "Duplicate examples waste compute and can bias the model.",
        );
    } else if report.metrics.duplication_rate > 0.01 {
        report.add(
            "Duplication",
            Severity::Warn,
            &format!("{:.1}% duplicate outputs", report.metrics.duplication_rate * 100.0),
            "",
        );
    } else {
        report.add(
            "Duplication",
            Severity::Info,
            &format!("{:.1}% duplication rate", report.metrics.duplication_rate * 100.0),
            "OK",
        );
    }

    // Self-BLEU (template detection)
    let self_bleu = compute_self_bleu(&non_empty, 1000, 4);
    report.metrics.self_bleu = self_bleu;
    if let Some(sb) = self_bleu {
        let threshold = if matches!(task_type, "creative" | "generate" | "generation") {
            0.3
        } else {
            0.5
        };
        let concern = if matches!(task_type, "creative" | "generate" | "generation") {
            0.6
        } else {
            0.7
        };
        if sb > concern {
            report.add(
                "Self-BLEU",
                Severity::Warn,
                &format!("Self-BLEU = {:.3} (HIGH). Strong template contamination signal.", sb),
                "Lesson AC-v2: Templated data (self-BLEU > 0.6) teaches templates, not generation.",
            );
        } else if sb > threshold {
            report.add(
                "Self-BLEU",
                Severity::Warn,
                &format!("Self-BLEU = {:.3} (concerning). Some repetitive patterns detected.", sb),
                "",
            );
        } else {
            report.add(
                "Self-BLEU",
                Severity::Info,
                &format!("Self-BLEU = {:.3} (good diversity)", sb),
                "OK",
            );
        }
    }

    report
}

// ---------------------------------------------------------------------------
// Self-BLEU implementation (no external dependencies)
// ---------------------------------------------------------------------------

fn compute_self_bleu(texts: &[&str], n_samples: usize, ngram: usize) -> Option<f64> {
    if texts.len() < 10 {
        return None;
    }

    let sample_count = n_samples.min(texts.len());
    let step = texts.len() as f64 / sample_count as f64;

    // Owned tokenized strings (lowercased)
    let tokenized: Vec<Vec<String>> = (0..sample_count)
        .map(|i| {
            let idx = (i as f64 * step) as usize;
            texts[idx]
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect()
        })
        .collect();

    let score_count = 100.min(tokenized.len());
    let ref_count = 50.min(tokenized.len().saturating_sub(1));
    if ref_count == 0 {
        return None;
    }

    let mut scores = Vec::with_capacity(score_count);
    for i in 0..score_count {
        let hypothesis: Vec<&str> = tokenized[i].iter().map(|s| s.as_str()).collect();
        let mut refs: Vec<Vec<&str>> = Vec::with_capacity(ref_count);
        let mut count = 0;
        for j in 0..tokenized.len() {
            if j == i {
                continue;
            }
            refs.push(tokenized[j].iter().map(|s| s.as_str()).collect());
            count += 1;
            if count >= ref_count {
                break;
            }
        }
        scores.push(simple_bleu(&hypothesis, &refs, ngram));
    }

    if scores.is_empty() {
        None
    } else {
        Some(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

fn simple_bleu(hypothesis: &[&str], references: &[Vec<&str>], max_n: usize) -> f64 {
    if hypothesis.is_empty() {
        return 0.0;
    }

    let mut scores = Vec::new();
    for n in 1..=max_n {
        let hyp_ngrams = get_ngrams(hypothesis, n);
        if hyp_ngrams.is_empty() {
            break;
        }

        let mut ref_ngrams: HashMap<Vec<&str>, usize> = HashMap::new();
        for reference in references {
            for (ng, count) in get_ngrams(reference, n) {
                let entry = ref_ngrams.entry(ng).or_insert(0);
                *entry = (*entry).max(count);
            }
        }

        let mut matches = 0usize;
        let mut total = 0usize;
        for (ng, count) in &hyp_ngrams {
            let ref_count = ref_ngrams.get(ng.as_slice()).copied().unwrap_or(0);
            matches += (*count).min(ref_count);
            total += count;
        }

        if total > 0 {
            scores.push(matches as f64 / total as f64);
        } else {
            scores.push(0.0);
        }
    }

    if scores.is_empty() {
        return 0.0;
    }

    // Geometric mean
    let mut product = 1.0;
    for &s in &scores {
        product *= s.max(1e-10);
    }
    product.powf(1.0 / scores.len() as f64)
}

fn get_ngrams<'a>(tokens: &'a [&'a str], n: usize) -> HashMap<Vec<&'a str>, usize> {
    let mut counts = HashMap::new();
    if tokens.len() < n {
        return counts;
    }
    for i in 0..=tokens.len() - n {
        let ng = tokens[i..i + n].to_vec();
        *counts.entry(ng).or_insert(0) += 1;
    }
    counts
}

/// Simple hash for dedup (not cryptographic, just for equality).
fn md5_hash(s: &str) -> u128 {
    // Simple FNV-like hash, sufficient for dedup detection.
    let mut h: u128 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u128;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
