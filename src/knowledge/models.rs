/// Known model specifications.

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    pub params: &'static str,
    pub vram_bf16: &'static str,
    pub vram_qlora: &'static str,
    pub max_seq_len: u32,
    pub architecture: &'static str,
    pub good_for: &'static [&'static str],
}

pub static MODELS: &[(&str, ModelSpec)] = &[
    (
        "mistral-7b",
        ModelSpec {
            name: "Mistral-7B-Instruct",
            params: "7B",
            vram_bf16: "14GB",
            vram_qlora: "6GB",
            max_seq_len: 32768,
            architecture: "decoder-only",
            good_for: &["generation", "instruction-following"],
        },
    ),
    (
        "llama-3-8b",
        ModelSpec {
            name: "Llama-3-8B",
            params: "8B",
            vram_bf16: "16GB",
            vram_qlora: "6GB",
            max_seq_len: 8192,
            architecture: "decoder-only",
            good_for: &["generation", "instruction-following", "reasoning"],
        },
    ),
    (
        "flan-t5-xl",
        ModelSpec {
            name: "Flan-T5-XL",
            params: "3B",
            vram_bf16: "6GB",
            vram_qlora: "3GB",
            max_seq_len: 512,
            architecture: "encoder-decoder",
            good_for: &["classification", "structured-generation", "summarization"],
        },
    ),
    (
        "flan-t5-xxl",
        ModelSpec {
            name: "Flan-T5-XXL",
            params: "11B",
            vram_bf16: "22GB",
            vram_qlora: "8GB",
            max_seq_len: 512,
            architecture: "encoder-decoder",
            good_for: &["classification", "structured-generation", "summarization"],
        },
    ),
    (
        "qwen-2.5-7b",
        ModelSpec {
            name: "Qwen-2.5-7B",
            params: "7B",
            vram_bf16: "14GB",
            vram_qlora: "6GB",
            max_seq_len: 32768,
            architecture: "decoder-only",
            good_for: &["generation", "multilingual", "reasoning"],
        },
    ),
    (
        "phi-3-mini",
        ModelSpec {
            name: "Phi-3-Mini-4K",
            params: "3.8B",
            vram_bf16: "8GB",
            vram_qlora: "4GB",
            max_seq_len: 4096,
            architecture: "decoder-only",
            good_for: &["generation", "reasoning", "code"],
        },
    ),
    (
        "gemma-2-9b",
        ModelSpec {
            name: "Gemma-2-9B",
            params: "9B",
            vram_bf16: "18GB",
            vram_qlora: "7GB",
            max_seq_len: 8192,
            architecture: "decoder-only",
            good_for: &["generation", "instruction-following", "reasoning"],
        },
    ),
    (
        "gemma-2-2b",
        ModelSpec {
            name: "Gemma-2-2B",
            params: "2B",
            vram_bf16: "4GB",
            vram_qlora: "2GB",
            max_seq_len: 8192,
            architecture: "decoder-only",
            good_for: &["generation", "classification", "lightweight"],
        },
    ),
    (
        "qwen-2.5-14b",
        ModelSpec {
            name: "Qwen-2.5-14B",
            params: "14B",
            vram_bf16: "28GB",
            vram_qlora: "10GB",
            max_seq_len: 32768,
            architecture: "decoder-only",
            good_for: &["generation", "multilingual", "reasoning", "code"],
        },
    ),
    (
        "qwen-2.5-3b",
        ModelSpec {
            name: "Qwen-2.5-3B",
            params: "3B",
            vram_bf16: "6GB",
            vram_qlora: "3GB",
            max_seq_len: 32768,
            architecture: "decoder-only",
            good_for: &["generation", "multilingual", "lightweight"],
        },
    ),
    (
        "deepseek-v2-lite",
        ModelSpec {
            name: "DeepSeek-V2-Lite",
            params: "16B",
            vram_bf16: "32GB",
            vram_qlora: "12GB",
            max_seq_len: 32768,
            architecture: "decoder-only",
            good_for: &["generation", "reasoning", "code", "math"],
        },
    ),
    (
        "llama-3.1-70b",
        ModelSpec {
            name: "Llama-3.1-70B",
            params: "70B",
            vram_bf16: "140GB",
            vram_qlora: "40GB",
            max_seq_len: 131072,
            architecture: "decoder-only",
            good_for: &["generation", "reasoning", "instruction-following"],
        },
    ),
    (
        "phi-3-medium",
        ModelSpec {
            name: "Phi-3-Medium-14B",
            params: "14B",
            vram_bf16: "28GB",
            vram_qlora: "10GB",
            max_seq_len: 4096,
            architecture: "decoder-only",
            good_for: &["generation", "reasoning", "code"],
        },
    ),
    (
        "mistral-nemo-12b",
        ModelSpec {
            name: "Mistral-Nemo-12B",
            params: "12B",
            vram_bf16: "24GB",
            vram_qlora: "8GB",
            max_seq_len: 128000,
            architecture: "decoder-only",
            good_for: &["generation", "instruction-following", "multilingual"],
        },
    ),
];

/// Find a model by key or fuzzy name match.
pub fn find_model(name: &str) -> Option<&'static (&'static str, ModelSpec)> {
    let name_lower = name.to_lowercase().replace(['-', '_'], "");
    MODELS.iter().find(|(key, _)| {
        let key_clean = key.replace('-', "");
        name_lower.contains(&key_clean) || key_clean.contains(&name_lower)
    })
}

/// Parse a human-readable VRAM string like "14GB" to f64.
pub fn parse_vram_gb(s: &str) -> f64 {
    s.to_uppercase()
        .replace("GB", "")
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
}

/// Parse a human-readable param count like "7B", "125M" to u64.
pub fn parse_param_count(s: &str) -> u64 {
    let s = s.trim().to_uppercase();
    if let Some(n) = s.strip_suffix('B') {
        (n.parse::<f64>().unwrap_or(0.0) * 1e9) as u64
    } else if let Some(n) = s.strip_suffix('M') {
        (n.parse::<f64>().unwrap_or(0.0) * 1e6) as u64
    } else if let Some(n) = s.strip_suffix('K') {
        (n.parse::<f64>().unwrap_or(0.0) * 1e3) as u64
    } else {
        s.parse::<u64>().unwrap_or(0)
    }
}
