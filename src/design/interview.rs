/// Design interview: gather requirements for architecture recommendation.
/// In Rust we skip interactive prompts — this is data-only + file loading.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Classify,
    Generate,
    Embed,
    Other,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Classify => "classify",
            TaskType::Generate => "generate",
            TaskType::Embed => "embed",
            TaskType::Other => "other",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "classify" => TaskType::Classify,
            "generate" => TaskType::Generate,
            "embed" => TaskType::Embed,
            _ => TaskType::Other,
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputFormat {
    Text,
    Structured,
    Multimodal,
}

impl InputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            InputFormat::Text => "text",
            InputFormat::Structured => "structured",
            InputFormat::Multimodal => "multimodal",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "structured" => InputFormat::Structured,
            "multimodal" => InputFormat::Multimodal,
            _ => InputFormat::Text,
        }
    }
}

impl std::fmt::Display for InputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Classes,
    Text,
    Embeddings,
}

impl OutputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::Classes => "classes",
            OutputFormat::Text => "text",
            OutputFormat::Embeddings => "embeddings",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "classes" => OutputFormat::Classes,
            "embeddings" => OutputFormat::Embeddings,
            _ => OutputFormat::Text,
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingApproach {
    FineTune,
    Scratch,
}

impl TrainingApproach {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrainingApproach::FineTune => "fine-tune",
            TrainingApproach::Scratch => "scratch",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().replace('-', "_").as_str() {
            "scratch" => TrainingApproach::Scratch,
            _ => TrainingApproach::FineTune,
        }
    }
}

impl std::fmt::Display for TrainingApproach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMethod {
    Full,
    Lora,
    Qlora,
}

impl TrainingMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrainingMethod::Full => "full",
            TrainingMethod::Lora => "lora",
            TrainingMethod::Qlora => "qlora",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lora" => TrainingMethod::Lora,
            "qlora" => TrainingMethod::Qlora,
            _ => TrainingMethod::Full,
        }
    }
}

impl std::fmt::Display for TrainingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreativityPriority {
    Creativity,
    Consistency,
    Balanced,
}

impl CreativityPriority {
    pub fn as_str(&self) -> &'static str {
        match self {
            CreativityPriority::Creativity => "creativity",
            CreativityPriority::Consistency => "consistency",
            CreativityPriority::Balanced => "balanced",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "creativity" => CreativityPriority::Creativity,
            "consistency" => CreativityPriority::Consistency,
            _ => CreativityPriority::Balanced,
        }
    }
}

impl std::fmt::Display for CreativityPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// All inputs gathered from the design interview (or loaded from YAML).
#[derive(Debug, Clone)]
pub struct DesignInputs {
    // Core
    pub task_description: String,
    pub task_type: TaskType,
    pub input_format: InputFormat,
    pub output_format: OutputFormat,
    pub approach: TrainingApproach,

    // Fine-tuning specific
    pub base_model: String,
    pub training_method: TrainingMethod,
    pub vram_gb: f64,
    pub data_size: usize,
    pub creativity: CreativityPriority,

    // Scratch specific
    pub param_budget: String,
    pub input_seq_len: usize,
    pub output_seq_len: usize,
    pub latency_ms: f64,
}

impl Default for DesignInputs {
    fn default() -> Self {
        Self {
            task_description: String::new(),
            task_type: TaskType::Generate,
            input_format: InputFormat::Text,
            output_format: OutputFormat::Text,
            approach: TrainingApproach::FineTune,
            base_model: String::new(),
            training_method: TrainingMethod::Lora,
            vram_gb: 0.0,
            data_size: 0,
            creativity: CreativityPriority::Balanced,
            param_budget: String::new(),
            input_seq_len: 512,
            output_seq_len: 256,
            latency_ms: 0.0,
        }
    }
}

impl DesignInputs {
    /// Convert to a serde_yaml::Value for serialization.
    pub fn to_yaml_value(&self) -> serde_yaml::Value {
        let mut map = serde_yaml::Mapping::new();
        let s = |v: &str| serde_yaml::Value::String(v.to_string());
        let n = |v: f64| serde_yaml::Value::Number(serde_yaml::Number::from(v));

        if !self.task_description.is_empty() {
            map.insert(s("task_description"), s(&self.task_description));
        }
        map.insert(s("task_type"), s(self.task_type.as_str()));
        map.insert(s("input_format"), s(self.input_format.as_str()));
        map.insert(s("output_format"), s(self.output_format.as_str()));
        map.insert(s("approach"), s(self.approach.as_str()));

        if !self.base_model.is_empty() {
            map.insert(s("base_model"), s(&self.base_model));
        }
        map.insert(s("training_method"), s(self.training_method.as_str()));
        if self.vram_gb > 0.0 {
            map.insert(s("vram_gb"), n(self.vram_gb));
        }
        if self.data_size > 0 {
            map.insert(
                s("data_size"),
                serde_yaml::Value::Number(serde_yaml::Number::from(self.data_size as u64)),
            );
        }
        map.insert(s("creativity"), s(self.creativity.as_str()));

        if !self.param_budget.is_empty() {
            map.insert(s("param_budget"), s(&self.param_budget));
        }
        map.insert(
            s("input_seq_len"),
            serde_yaml::Value::Number(serde_yaml::Number::from(self.input_seq_len as u64)),
        );
        map.insert(
            s("output_seq_len"),
            serde_yaml::Value::Number(serde_yaml::Number::from(self.output_seq_len as u64)),
        );

        serde_yaml::Value::Mapping(map)
    }

    /// Load from a generic key-value map (parsed from YAML).
    pub fn from_map(m: &HashMap<String, serde_yaml::Value>) -> Self {
        let gs = |m: &HashMap<String, serde_yaml::Value>, k: &str| -> String {
            m.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let gf = |m: &HashMap<String, serde_yaml::Value>, k: &str| -> f64 {
            m.get(k).and_then(|v| match v {
                serde_yaml::Value::Number(n) => n.as_f64(),
                serde_yaml::Value::String(s) => s.parse().ok(),
                _ => None,
            }).unwrap_or(0.0)
        };
        let gu = |m: &HashMap<String, serde_yaml::Value>, k: &str| -> usize {
            m.get(k).and_then(|v| match v {
                serde_yaml::Value::Number(n) => n.as_u64().map(|x| x as usize),
                serde_yaml::Value::String(s) => s.parse().ok(),
                _ => None,
            }).unwrap_or(0)
        };

        Self {
            task_description: gs(m, "task_description"),
            task_type: TaskType::from_str_loose(&gs(m, "task_type")),
            input_format: InputFormat::from_str_loose(&gs(m, "input_format")),
            output_format: OutputFormat::from_str_loose(&gs(m, "output_format")),
            approach: TrainingApproach::from_str_loose(&gs(m, "approach")),
            base_model: gs(m, "base_model"),
            training_method: TrainingMethod::from_str_loose(&gs(m, "training_method")),
            vram_gb: gf(m, "vram_gb"),
            data_size: gu(m, "data_size"),
            creativity: CreativityPriority::from_str_loose(&gs(m, "creativity")),
            param_budget: gs(m, "param_budget"),
            input_seq_len: {
                let v = gu(m, "input_seq_len");
                if v == 0 { 512 } else { v }
            },
            output_seq_len: {
                let v = gu(m, "output_seq_len");
                if v == 0 { 256 } else { v }
            },
            latency_ms: gf(m, "latency_ms"),
        }
    }
}

/// Load interview answers from a YAML file.
pub fn load_interview_from_file(path: &Path) -> Result<DesignInputs, Error> {
    let text = std::fs::read_to_string(path).map_err(Error::Io)?;
    let raw: HashMap<String, serde_yaml::Value> =
        serde_yaml::from_str(&text).map_err(Error::Yaml)?;
    Ok(DesignInputs::from_map(&raw))
}

/// Save interview answers to a YAML file.
pub fn save_interview(inputs: &DesignInputs, path: &Path) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let val = inputs.to_yaml_value();
    let text = serde_yaml::to_string(&val).map_err(Error::Yaml)?;
    std::fs::write(path, text).map_err(Error::Io)?;
    Ok(())
}
