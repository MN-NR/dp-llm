# ε=3, non-DP finetuning, and unfinetuned base

The non-DP `ε=∞` values are the `baseline` rows from `results_testing/results_table.csv`. They predate the July 24, 2026 test-data rebuild; therefore cross-column deltas involving ε=∞ are provisional rather than strictly matched.

| Model           | Metric                  |   Unfinetuned base (current) |   Non-DP finetuned ε=∞ (CSV) |   DP finetuned ε=3 (current) |   Δ ε=∞ vs Base |   Δ ε=3 vs Base |   Δ ε=3 vs ε=∞ |
|:----------------|:------------------------|-----------------------------:|-----------------------------:|-----------------------------:|----------------:|----------------:|---------------:|
| Gemma 2B IT     | Classification accuracy |                       0.7830 |                       0.8830 |                       0.7930 |          0.1000 |          0.0100 |        -0.0900 |
| Gemma 2B IT     | NLI accuracy            |                       0.6100 |                       0.8320 |                       0.7130 |          0.2220 |          0.1030 |        -0.1190 |
| Gemma 2B IT     | PII F1                  |                       0.6557 |                       0.9860 |                       0.7227 |          0.3303 |          0.0670 |        -0.2633 |
| VaultGemma      | Classification accuracy |                       0.2456 |                       0.2780 |                       0.2425 |          0.0324 |         -0.0031 |        -0.0355 |
| VaultGemma      | NLI accuracy            |                       0.3571 |                       0.3440 |                       0.3663 |         -0.0131 |          0.0092 |         0.0223 |
| VaultGemma      | PII F1                  |                       0.0000 |                       0.6463 |                       0.1659 |          0.6463 |          0.1659 |        -0.4804 |
| Gemma 3 1B      | Classification accuracy |                       0.6970 |                       0.8620 |                       0.6470 |          0.1650 |         -0.0500 |        -0.2150 |
| Gemma 3 1B      | NLI accuracy            |                       0.5620 |                       0.6160 |                       0.5010 |          0.0540 |         -0.0610 |        -0.1150 |
| Gemma 3 1B      | PII F1                  |                       0.3508 |                       0.9331 |                       0.6198 |          0.5823 |          0.2690 |        -0.3133 |
| Llama 3.2 1B IT | Classification accuracy |                       0.2860 |                       0.8950 |                       0.6810 |          0.6090 |          0.3950 |        -0.2140 |
| Llama 3.2 1B IT | NLI accuracy            |                       0.3640 |                       0.7990 |                       0.4580 |          0.4350 |          0.0940 |        -0.3410 |
| Llama 3.2 1B IT | PII F1                  |                       0.0000 |                       0.9734 |                       0.5681 |          0.9734 |          0.5681 |        -0.4053 |
