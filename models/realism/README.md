# Optional DistilKoBERT Model

Place the private fine-tuned checkpoint at this exact local path:

```text
models/realism/best_student.pt
```

`best_student.pt` enables the R-axis voice-phishing realism evaluation. The
checkpoint itself is excluded from Git. If it is absent, the pipeline
automatically runs with KEV A-D only.
