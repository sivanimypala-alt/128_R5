# Model validation note

A chronological holdout was run on the public labelled sample: labels dated before 2026-06-01 were used for training (2,043 records), and labels dated 2026-06-01 onward were held out (957 records). The held-out results were:

- Issue type macro-F1: **0.8868**
- Severity weighted-F1: **0.9155**
- Actionable F1: **0.9277**

These are local validation results on a small public labelled sample, not the official hidden-test score. The public labels are deliberately over-sampled by category, and repeated text patterns may make this split easier than genuinely new feedback. The official evaluator should still be run after each model change.
