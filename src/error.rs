/// Unified error type for TransXLab.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Config validation: {0}")]
    ConfigValidation(String),

    #[error("Data validation: {0}")]
    DataValidation(String),

    #[error("Probe error: {0}")]
    Probe(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
