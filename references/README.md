# Local References

This directory is the fixed location for optional local-only pipeline references.
The reference contents are intentionally excluded from Git; only this placement guide is tracked.

```text
references/
├── Tactic.pdf                    # Generation: attack objective and progression
├── Technique.pdf                 # Generation: concrete methods that realize a tactic
├── Scenario.pdf                  # Generation/KEV: existing-scenario comparison context
├── positive_format_refs.jsonl    # Optional generation-style reference only
└── README.md
```

Every reference is optional. The pipeline skips any missing PDF. If
`positive_format_refs.jsonl` is absent, dialogue generation runs without private
format examples. Do not commit reference contents or personal/raw call data.
