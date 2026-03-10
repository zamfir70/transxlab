/// Hard-won lessons from real training failures.

#[derive(Debug, Clone)]
pub struct Finding {
    pub key: &'static str,
    pub message: &'static str,
}

#[derive(Debug, Clone)]
pub struct Lesson {
    pub id: &'static str,
    pub cost: &'static str,
    pub summary: &'static str,
    pub findings: &'static [Finding],
    pub rules_derived: &'static [&'static str],
}

pub static LESSONS: &[Lesson] = &[Lesson {
    id: "AC-v2",
    cost: "$665",
    summary: "Abductive Completor v2 fine-tuning failures",
    findings: &[
        Finding {
            key: "lr",
            message: "1e-4 is too high for fine-tuning Flan-T5-XL. Use 1e-5 to 5e-5.",
        },
        Finding {
            key: "diversity_loss",
            message: "For creative generation, diversity_loss_weight=0.0 causes mode collapse. Use >= 0.3.",
        },
        Finding {
            key: "monitoring",
            message: "val_loss alone is insufficient. Must eval generation quality on novel queries.",
        },
        Finding {
            key: "data",
            message: "Templated data (self-BLEU > 0.6) teaches templates, not generation. Use distillation.",
        },
    ],
    rules_derived: &[
        "LR Too High (Fine-tune)",
        "Template Memorization",
        "Missing Diversity Signal",
        "Wrong Eval Metric",
    ],
}];

/// Find the lesson that derived a given failure rule.
pub fn find_lesson_for_rule(rule_name: &str) -> Option<&'static Lesson> {
    LESSONS
        .iter()
        .find(|l| l.rules_derived.contains(&rule_name))
}

/// Get a specific finding from a lesson by key.
pub fn get_finding(lesson: &Lesson, key: &str) -> Option<&'static str> {
    lesson.findings.iter().find(|f| f.key == key).map(|f| f.message)
}
