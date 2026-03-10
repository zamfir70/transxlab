/// HuggingFace Hub integration: fetch model config.json for auto-detection.
use crate::knowledge::models::ModelSpec;

#[derive(Debug, Clone)]
pub struct HubModelInfo {
    pub repo_id: String,
    pub architecture: String,
    pub param_count: Option<u64>,
    pub max_position_embeddings: Option<u32>,
    pub hidden_size: Option<u32>,
    pub num_layers: Option<u32>,
    pub torch_dtype: Option<String>,
}

impl HubModelInfo {
    /// Convert to a ModelSpec for use in the rest of the system.
    pub fn to_model_spec(&self) -> ModelSpec {
        let params = match self.param_count {
            Some(n) if n >= 1_000_000_000 => format!("{}B", n / 1_000_000_000),
            Some(n) if n >= 1_000_000 => format!("{}M", n / 1_000_000),
            Some(n) => format!("{}", n),
            None => estimate_params_from_config(self),
        };

        let vram_bf16 = match self.param_count {
            Some(n) => format!("{}GB", (n as f64 * 2.0 / 1e9).ceil() as u64),
            None => "unknown".to_string(),
        };

        let vram_qlora = match self.param_count {
            Some(n) => format!("{}GB", ((n as f64 * 0.5 + 1e9) / 1e9).ceil() as u64),
            None => "unknown".to_string(),
        };

        let max_seq = self.max_position_embeddings.unwrap_or(2048);

        // Leak strings so we can return &'static str (these are one-off allocations)
        let name: &'static str = Box::leak(self.repo_id.clone().into_boxed_str());
        let params: &'static str = Box::leak(params.into_boxed_str());
        let vram_bf16: &'static str = Box::leak(vram_bf16.into_boxed_str());
        let vram_qlora: &'static str = Box::leak(vram_qlora.into_boxed_str());
        let arch: &'static str = Box::leak(self.architecture.clone().into_boxed_str());

        ModelSpec {
            name,
            params,
            vram_bf16,
            vram_qlora,
            max_seq_len: max_seq,
            architecture: arch,
            good_for: &[],
        }
    }
}

fn estimate_params_from_config(info: &HubModelInfo) -> String {
    match (info.hidden_size, info.num_layers) {
        (Some(h), Some(l)) => {
            // Rough: params ≈ 12 * L * H^2 (transformer parameter formula)
            let est = 12.0 * l as f64 * (h as f64).powi(2);
            if est >= 1e9 {
                format!("{:.0}B", est / 1e9)
            } else if est >= 1e6 {
                format!("{:.0}M", est / 1e6)
            } else {
                format!("{:.0}", est)
            }
        }
        _ => "unknown".to_string(),
    }
}

/// Fetch config.json from HuggingFace Hub for a public model.
///
/// `repo_id` should be like "meta-llama/Llama-3-8B" or "mistralai/Mistral-7B-v0.1".
pub fn fetch_hub_config(repo_id: &str) -> Result<HubModelInfo, String> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/config.json",
        repo_id
    );

    let response = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .map_err(|e| format!("Failed to fetch {}: {}", url, e))?;

    let reader = response.into_reader();
    let body: serde_json::Value = serde_json::from_reader(reader)
        .map_err(|e| format!("Failed to parse config.json: {}", e))?;

    let arch = extract_architecture(&body);
    let param_count = extract_param_count(&body);
    let max_pos = body
        .get("max_position_embeddings")
        .and_then(|v: &serde_json::Value| v.as_u64())
        .map(|v| v as u32);
    let hidden = body
        .get("hidden_size")
        .and_then(|v: &serde_json::Value| v.as_u64())
        .map(|v| v as u32);
    let layers = body
        .get("num_hidden_layers")
        .or_else(|| body.get("num_layers"))
        .or_else(|| body.get("n_layer"))
        .and_then(|v: &serde_json::Value| v.as_u64())
        .map(|v| v as u32);
    let dtype = body
        .get("torch_dtype")
        .and_then(|v: &serde_json::Value| v.as_str())
        .map(|s: &str| s.to_string());

    Ok(HubModelInfo {
        repo_id: repo_id.to_string(),
        architecture: arch,
        param_count,
        max_position_embeddings: max_pos,
        hidden_size: hidden,
        num_layers: layers,
        torch_dtype: dtype,
    })
}

fn extract_architecture(config: &serde_json::Value) -> String {
    // HF config.json uses "architectures" array or "model_type"
    if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
        if let Some(first) = archs.first().and_then(|v| v.as_str()) {
            // Check for encoder-decoder patterns
            let lower = first.to_lowercase();
            if lower.contains("seq2seq")
                || lower.contains("conditionalgener")
                || lower.contains("t5")
                || lower.contains("bart")
                || lower.contains("encoder_decoder")
            {
                return "encoder-decoder".to_string();
            }
            return "decoder-only".to_string();
        }
    }

    if let Some(mt) = config.get("model_type").and_then(|v| v.as_str()) {
        let lower = mt.to_lowercase();
        if lower.contains("t5")
            || lower.contains("bart")
            || lower.contains("marian")
            || lower.contains("pegasus")
        {
            return "encoder-decoder".to_string();
        }
    }

    // Check for explicit is_encoder_decoder flag
    if config
        .get("is_encoder_decoder")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return "encoder-decoder".to_string();
    }

    "decoder-only".to_string()
}

fn extract_param_count(config: &serde_json::Value) -> Option<u64> {
    // Some configs have explicit param count fields
    for key in &["num_parameters", "n_params", "total_params"] {
        if let Some(n) = config.get(*key).and_then(|v| v.as_u64()) {
            return Some(n);
        }
    }

    // Estimate from architecture if we have hidden_size and num_hidden_layers
    let hidden = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())? as f64;
    let layers = config
        .get("num_hidden_layers")
        .or_else(|| config.get("num_layers"))
        .or_else(|| config.get("n_layer"))
        .and_then(|v| v.as_u64())? as f64;
    let vocab = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as f64;
    let intermediate = config
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or((hidden * 4.0) as u64) as f64;

    // Transformer param estimate: embedding + layers*(attn + ffn) + lm_head
    let embedding = vocab * hidden;
    let attn_per_layer = 4.0 * hidden * hidden; // Q, K, V, O
    let ffn_per_layer = 2.0 * hidden * intermediate; // up + down
    let layer_norm = 2.0 * hidden; // per layer
    let total = embedding
        + layers * (attn_per_layer + ffn_per_layer + layer_norm)
        + embedding; // lm_head (usually tied)

    Some(total as u64)
}
