/// Data source suggestions based on task domain.

#[derive(Debug, Clone)]
pub struct DataSource {
    pub name: &'static str,
    pub domain: &'static str,
    pub description: &'static str,
    pub url: &'static str,
    pub notes: &'static str,
    pub warnings: &'static [&'static str],
}

static SOURCES: &[DataSource] = &[
    DataSource {
        name: "ConceptNet",
        domain: "relational",
        description: "Commonsense knowledge graph with relational data for entity pairs.",
        url: "conceptnet.io",
        notes: "Good for relational reasoning, entity pairs, commonsense.",
        warnings: &[],
    },
    DataSource {
        name: "WordNet",
        domain: "relational",
        description: "Lexical database with hypernyms, synonyms, and concept hierarchies.",
        url: "wordnet.princeton.edu",
        notes: "Good for concept expansion, word relationships.",
        warnings: &[],
    },
    DataSource {
        name: "Wikidata",
        domain: "relational",
        description: "Structured knowledge base from Wikimedia.",
        url: "wikidata.org",
        notes: "Massive, but needs filtering for relevance.",
        warnings: &["Raw dump is very large. Filter to relevant properties."],
    },
    DataSource {
        name: "HuggingFace Datasets",
        domain: "general",
        description: "Hub of ML datasets across many tasks and domains.",
        url: "huggingface.co/datasets",
        notes: "Search for task-specific datasets. Check licenses.",
        warnings: &[],
    },
    DataSource {
        name: "The Stack",
        domain: "code",
        description: "Large code dataset across many languages.",
        url: "huggingface.co/datasets/bigcode/the-stack",
        notes: "Good for code generation tasks.",
        warnings: &["Check license compatibility for your use case."],
    },
    DataSource {
        name: "S2ORC",
        domain: "scientific",
        description: "Semantic Scholar Open Research Corpus. Full text of academic papers.",
        url: "github.com/allenai/s2orc",
        notes: "Good for scientific domain adaptation.",
        warnings: &["Large corpus. Filter to relevant fields."],
    },
    DataSource {
        name: "NLI Datasets (SNLI, MultiNLI)",
        domain: "classification",
        description: "Natural Language Inference datasets.",
        url: "huggingface.co/datasets/snli",
        notes: "Classification format. Popular for entailment tasks.",
        warnings: &[
            "HIGH RISK: NLI format creates template contamination.",
            "Lesson AC-v2: 15% NLI-format data caused template memorization.",
            "Do NOT use for generation tasks without heavy reformatting.",
        ],
    },
    DataSource {
        name: "Custom Distillation",
        domain: "any",
        description: "Prompt a large model (Claude, GPT-4, Qwen-32B) with your task to generate training data.",
        url: "",
        notes: "Most flexible. Quality depends on prompt engineering. Vary prompts and temperature for diversity.",
        warnings: &["Ensure diverse prompts to avoid self-BLEU > 0.3."],
    },
];

pub fn suggest_sources(task_type: &str, domain: &str) -> Vec<&'static DataSource> {
    let task_lower = task_type.to_lowercase();
    let domain_lower = domain.to_lowercase();

    let mut scored: Vec<(i32, &'static DataSource)> = Vec::new();

    for source in SOURCES.iter() {
        let mut relevance: i32 = 0;

        // Domain match
        if source.domain == "any" {
            relevance += 1;
        }
        if !domain_lower.is_empty() && domain_lower.contains(source.domain) {
            relevance += 3;
        }
        if !domain_lower.is_empty() && source.domain.contains(&domain_lower) {
            relevance += 3;
        }

        // Task match
        if task_lower.contains("generat") && (source.domain == "any" || source.domain == "general")
        {
            relevance += 2;
        }
        if task_lower.contains("classif") && source.domain == "classification" {
            relevance += 3;
        }
        if task_lower.contains("code") && source.domain == "code" {
            relevance += 3;
        }
        if task_lower.contains("scien") && source.domain == "scientific" {
            relevance += 3;
        }
        if task_lower.contains("relat") && source.domain == "relational" {
            relevance += 3;
        }
        if task_lower.contains("knowledge") && source.domain == "relational" {
            relevance += 2;
        }

        if relevance > 0 {
            scored.push((relevance, source));
        }
    }

    // Always include custom distillation
    let distill = SOURCES.iter().find(|s| s.name == "Custom Distillation").unwrap();
    if !scored.iter().any(|(_, s)| std::ptr::eq(*s, distill)) {
        scored.push((1, distill));
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().map(|(_, s)| s).collect()
}
