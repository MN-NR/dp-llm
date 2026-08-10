# All-model group privacy scaling

Generated: 2026-07-26T20:18:43-04:00

The basic group-privacy bound scales epsilon linearly: effective_epsilon = k × epsilon.
For descriptive comparison, empirical scaling is the canary memorization gap at k× divided by the 1× gap.
`consistent` means empirical/theoretical scaling is within ±20%; otherwise it is `exceeds` or `falls_short`.

Important: DP group privacy bounds privacy loss/odds, not cross-entropy loss gaps. The scaling assessment is therefore a descriptive diagnostic, not a test of the DP theorem.

## ε=3 and ε=64 assessment counts (10× and 50×)

- falls_short: 16
- consistent: 0
- exceeds: 0
- undefined: 0

## Focus table

| Model | ε | k | kε | Gap | Empirical scaling | Linear prediction | Empirical/theory | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| gemma3_1b | 3 | 10 | 30 | 0.00865 | 0.406 | 10 | 0.041 | falls_short |
| gemma3_1b | 3 | 50 | 150 | 0.07449 | 3.498 | 50 | 0.070 | falls_short |
| gemma3_1b | 64 | 10 | 640 | 0.00671 | 0.263 | 10 | 0.026 | falls_short |
| gemma3_1b | 64 | 50 | 3200 | 0.09381 | 3.684 | 50 | 0.074 | falls_short |
| gemma_2b_it | 3 | 10 | 30 | 0.00349 | 0.131 | 10 | 0.013 | falls_short |
| gemma_2b_it | 3 | 50 | 150 | 0.05956 | 2.239 | 50 | 0.045 | falls_short |
| gemma_2b_it | 64 | 10 | 640 | -0.00134 | -0.042 | 10 | -0.004 | falls_short |
| gemma_2b_it | 64 | 50 | 3200 | 0.07916 | 2.494 | 50 | 0.050 | falls_short |
| llama3_2_1b_it | 3 | 10 | 30 | 0.01894 | 1.270 | 10 | 0.127 | falls_short |
| llama3_2_1b_it | 3 | 50 | 150 | -0.00431 | -0.289 | 50 | -0.006 | falls_short |
| llama3_2_1b_it | 64 | 10 | 640 | 0.02229 | 1.232 | 10 | 0.123 | falls_short |
| llama3_2_1b_it | 64 | 50 | 3200 | 0.00532 | 0.294 | 50 | 0.006 | falls_short |
| vault_gemma | 3 | 10 | 30 | 0.00894 | 0.304 | 10 | 0.030 | falls_short |
| vault_gemma | 3 | 50 | 150 | 0.09796 | 3.332 | 50 | 0.067 | falls_short |
| vault_gemma | 64 | 10 | 640 | 0.01114 | 0.396 | 10 | 0.040 | falls_short |
| vault_gemma | 64 | 50 | 3200 | 0.10896 | 3.873 | 50 | 0.077 | falls_short |
