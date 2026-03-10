/// Data strategy recommender: suggest data approach based on task type.

#[derive(Debug, Clone)]
pub struct DataStrategy {
    pub name: &'static str,
    pub description: &'static str,
    pub minimum_examples: usize,
    pub recommended_examples: &'static str,
    pub format_notes: &'static str,
    pub quality_checks: &'static [&'static str],
    pub anti_patterns: &'static [&'static str],
    pub split_ratio: &'static str,
}

static DISTILLATION: DataStrategy = DataStrategy {
    name: "Distillation",
    description: "Use a large model (Qwen-32B, Claude, GPT-4) to generate training examples. \
                  The student model learns from the teacher's outputs.",
    minimum_examples: 2000,
    recommended_examples: "5K-10K",
    format_notes: "(input, high-quality output) pairs. Ensure diverse prompts to avoid template contamination.",
    quality_checks: &[
        "Self-BLEU < 0.3 (creative) or < 0.5 (structured)",
        "Output parse rate > 99% if structured",
        "Type/category diversity balanced",
    ],
    anti_patterns: &[
        "Using a single prompt template for all generation",
        "Not varying the teacher's temperature/sampling",
        "Generating more than 10 outputs per unique input",
    ],
    split_ratio: "90/5/5",
};

static LABELED_EXAMPLES: DataStrategy = DataStrategy {
    name: "Labeled Examples",
    description: "Human-labeled (input, output/class) pairs. Gold standard for classification. \
                  Can be crowd-sourced or expert-labeled.",
    minimum_examples: 500,
    recommended_examples: "1K-10K",
    format_notes: "Balanced classes. Stratified split to prevent class imbalance in val/test.",
    quality_checks: &[
        "Class balance: no class > 3x the smallest",
        "Inter-annotator agreement if crowd-sourced",
        "Edge case coverage",
    ],
    anti_patterns: &[
        "Imbalanced classes without addressing via sampling or loss weighting",
        "Label noise from low-quality annotation",
        "Annotation artifacts (spurious correlations with label)",
    ],
    split_ratio: "80/10/10",
};

static SELF_INSTRUCT: DataStrategy = DataStrategy {
    name: "Self-Instruct",
    description: "Generate instruction-response pairs using the model itself (or a seed set). \
                  Good for instruction following when no large teacher is available.",
    minimum_examples: 5000,
    recommended_examples: "10K-50K",
    format_notes: "(instruction, response) pairs. Include diverse task types.",
    quality_checks: &[
        "Instruction diversity (varied verbs, topics)",
        "Response quality sampling (manual review of 100+)",
        "No circular or nonsensical instructions",
    ],
    anti_patterns: &[
        "Using only 1-2 seed instruction types",
        "Not filtering low-quality generated pairs",
        "Instruction-response length correlation (always short-short or long-long)",
    ],
    split_ratio: "90/5/5",
};

static DOMAIN_CORPUS: DataStrategy = DataStrategy {
    name: "Domain Corpus",
    description: "Continued pretraining on domain-specific text. \
                  Adapts the model's language model to a new domain before task fine-tuning.",
    minimum_examples: 50000,
    recommended_examples: "100K-1M+ tokens",
    format_notes: "Raw text from the target domain. No labels needed.",
    quality_checks: &[
        "Corpus quality (no boilerplate, headers, or noise)",
        "Domain relevance (manual check of samples)",
        "Deduplication",
    ],
    anti_patterns: &[
        "Including off-topic content that dilutes domain signal",
        "Not deduplicating (web scrapes often have duplicates)",
        "Too small a corpus (< 1M tokens for continued pretraining)",
    ],
    split_ratio: "95/5",
};

static SYNTHETIC_STRUCTURED: DataStrategy = DataStrategy {
    name: "Synthetic Structured",
    description: "Programmatically generate examples with known structure. \
                  Good for structured output tasks where schema compliance matters.",
    minimum_examples: 1000,
    recommended_examples: "5K-50K",
    format_notes: "Schema-validated (input, structured output) pairs.",
    quality_checks: &[
        "Output parse rate = 100% (by construction)",
        "Input diversity (not all from same template)",
        "Edge case coverage (empty fields, max length, special chars)",
    ],
    anti_patterns: &[
        "Over-reliance on templates (vary generation logic)",
        "Not adding natural language variation to inputs",
        "Forgetting edge cases in the schema",
    ],
    split_ratio: "90/5/5",
};

pub fn recommend_strategy(task_type: &str, _creativity: &str) -> &'static DataStrategy {
    let task = task_type.to_lowercase();

    if task.contains("creative") || task.contains("generation") || task.contains("generate") {
        return &DISTILLATION;
    }
    if task.contains("classif") {
        return &LABELED_EXAMPLES;
    }
    if task.contains("instruct") {
        return &SELF_INSTRUCT;
    }
    if task.contains("domain") || task.contains("adapt") {
        return &DOMAIN_CORPUS;
    }
    if task.contains("structured") || task.contains("schema") {
        return &SYNTHETIC_STRUCTURED;
    }

    &DISTILLATION
}

pub fn get_all_strategies() -> Vec<(&'static str, &'static DataStrategy)> {
    vec![
        ("distillation", &DISTILLATION),
        ("labeled_examples", &LABELED_EXAMPLES),
        ("self_instruct", &SELF_INSTRUCT),
        ("domain_corpus", &DOMAIN_CORPUS),
        ("synthetic_structured", &SYNTHETIC_STRUCTURED),
    ]
}
