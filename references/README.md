# Local References

This directory is the fixed location for optional local-only pipeline references.
The reference contents are intentionally excluded from Git; only this placement guide is tracked.

```text
references/
├── Tactic.pdf                    # Generation: high-level attack tactics
├── Technique.pdf                 # Generation: technical/social-engineering techniques
├── Scenario.pdf                  # Generation and KEV evaluation: existing scenarios
├── positive_format_refs.jsonl    # Generation only: private positive-call format examples
└── README.md
```

Every reference is optional. The pipeline skips any missing PDF. If
`positive_format_refs.jsonl` is absent, dialogue generation runs without private
format examples. Do not commit reference contents or personal/raw call data.
