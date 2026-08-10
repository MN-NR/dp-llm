# Table 4 supplement: all DP epsilons versus base model

Δ vs Base = finetuned performance − base-model performance. Positive is better for accuracy/F1; negative is better for loss. Every base/finetuned pair uses the same current test records.

| Model          |       ε |    N | Metric                  |   Base |   Finetuned |   Δ vs Base | Direction   | Improves vs Base   |
|:---------------|--------:|-----:|:------------------------|-------:|------------:|------------:|:------------|:-------------------|
| gemma3_1b      |  3.0000 | 3000 | Average loss            | 3.2966 |      0.4439 |     -2.8527 | lower       | True               |
| gemma3_1b      |  3.0000 | 3000 | Classification accuracy | 0.6970 |      0.6470 |     -0.0500 | higher      | False              |
| gemma3_1b      |  3.0000 | 3000 | NLI accuracy            | 0.5620 |      0.5010 |     -0.0610 | higher      | False              |
| gemma3_1b      |  3.0000 | 3000 | PII F1                  | 0.3508 |      0.6198 |      0.2690 | higher      | True               |
| gemma3_1b      |  3.0000 | 3000 | PII relaxed F1          | 0.6338 |      0.7182 |      0.0844 | higher      | True               |
| gemma3_1b      |  5.0000 | 3000 | Average loss            | 3.2966 |      0.4435 |     -2.8531 | lower       | True               |
| gemma3_1b      |  5.0000 | 3000 | Classification accuracy | 0.6970 |      0.5950 |     -0.1020 | higher      | False              |
| gemma3_1b      |  5.0000 | 3000 | NLI accuracy            | 0.5620 |      0.5070 |     -0.0550 | higher      | False              |
| gemma3_1b      |  5.0000 | 3000 | PII F1                  | 0.3508 |      0.6180 |      0.2672 | higher      | True               |
| gemma3_1b      |  5.0000 | 3000 | PII relaxed F1          | 0.6338 |      0.7265 |      0.0927 | higher      | True               |
| gemma3_1b      |  8.0000 | 3000 | Average loss            | 3.2966 |      0.4269 |     -2.8696 | lower       | True               |
| gemma3_1b      |  8.0000 | 3000 | Classification accuracy | 0.6970 |      0.5570 |     -0.1400 | higher      | False              |
| gemma3_1b      |  8.0000 | 3000 | NLI accuracy            | 0.5620 |      0.4830 |     -0.0790 | higher      | False              |
| gemma3_1b      |  8.0000 | 3000 | PII F1                  | 0.3508 |      0.6154 |      0.2646 | higher      | True               |
| gemma3_1b      |  8.0000 | 3000 | PII relaxed F1          | 0.6338 |      0.7203 |      0.0865 | higher      | True               |
| gemma3_1b      | 16.0000 | 3000 | Average loss            | 3.2966 |      0.3889 |     -2.9076 | lower       | True               |
| gemma3_1b      | 16.0000 | 3000 | Classification accuracy | 0.6970 |      0.5310 |     -0.1660 | higher      | False              |
| gemma3_1b      | 16.0000 | 3000 | NLI accuracy            | 0.5620 |      0.5100 |     -0.0520 | higher      | False              |
| gemma3_1b      | 16.0000 | 3000 | PII F1                  | 0.3508 |      0.6254 |      0.2746 | higher      | True               |
| gemma3_1b      | 16.0000 | 3000 | PII relaxed F1          | 0.6338 |      0.7213 |      0.0875 | higher      | True               |
| gemma3_1b      | 64.0000 | 3000 | Average loss            | 3.2966 |      0.3087 |     -2.9878 | lower       | True               |
| gemma3_1b      | 64.0000 | 3000 | Classification accuracy | 0.6970 |      0.5050 |     -0.1920 | higher      | False              |
| gemma3_1b      | 64.0000 | 3000 | NLI accuracy            | 0.5620 |      0.5210 |     -0.0410 | higher      | False              |
| gemma3_1b      | 64.0000 | 3000 | PII F1                  | 0.3508 |      0.6400 |      0.2892 | higher      | True               |
| gemma3_1b      | 64.0000 | 3000 | PII relaxed F1          | 0.6338 |      0.7191 |      0.0853 | higher      | True               |
| gemma_2b_it    |  3.0000 | 3000 | Average loss            | 7.9478 |      0.3063 |     -7.6415 | lower       | True               |
| gemma_2b_it    |  3.0000 | 3000 | Classification accuracy | 0.7830 |      0.7930 |      0.0100 | higher      | True               |
| gemma_2b_it    |  3.0000 | 3000 | NLI accuracy            | 0.6100 |      0.7130 |      0.1030 | higher      | True               |
| gemma_2b_it    |  3.0000 | 3000 | PII F1                  | 0.6557 |      0.7227 |      0.0670 | higher      | True               |
| gemma_2b_it    |  3.0000 | 3000 | PII relaxed F1          | 0.7433 |      0.8028 |      0.0595 | higher      | True               |
| gemma_2b_it    |  5.0000 | 3000 | Average loss            | 7.9478 |      0.2952 |     -7.6526 | lower       | True               |
| gemma_2b_it    |  5.0000 | 3000 | Classification accuracy | 0.7830 |      0.7930 |      0.0100 | higher      | True               |
| gemma_2b_it    |  5.0000 | 3000 | NLI accuracy            | 0.6100 |      0.7140 |      0.1040 | higher      | True               |
| gemma_2b_it    |  5.0000 | 3000 | PII F1                  | 0.6557 |      0.7391 |      0.0834 | higher      | True               |
| gemma_2b_it    |  5.0000 | 3000 | PII relaxed F1          | 0.7433 |      0.8169 |      0.0736 | higher      | True               |
| gemma_2b_it    |  8.0000 | 3000 | Average loss            | 7.9478 |      0.2737 |     -7.6741 | lower       | True               |
| gemma_2b_it    |  8.0000 | 3000 | Classification accuracy | 0.7830 |      0.7940 |      0.0110 | higher      | True               |
| gemma_2b_it    |  8.0000 | 3000 | NLI accuracy            | 0.6100 |      0.7170 |      0.1070 | higher      | True               |
| gemma_2b_it    |  8.0000 | 3000 | PII F1                  | 0.6557 |      0.7415 |      0.0858 | higher      | True               |
| gemma_2b_it    |  8.0000 | 3000 | PII relaxed F1          | 0.7433 |      0.8221 |      0.0788 | higher      | True               |
| gemma_2b_it    | 16.0000 | 3000 | Average loss            | 7.9478 |      0.2416 |     -7.7062 | lower       | True               |
| gemma_2b_it    | 16.0000 | 3000 | Classification accuracy | 0.7830 |      0.8000 |      0.0170 | higher      | True               |
| gemma_2b_it    | 16.0000 | 3000 | NLI accuracy            | 0.6100 |      0.7080 |      0.0980 | higher      | True               |
| gemma_2b_it    | 16.0000 | 3000 | PII F1                  | 0.6557 |      0.7371 |      0.0814 | higher      | True               |
| gemma_2b_it    | 16.0000 | 3000 | PII relaxed F1          | 0.7433 |      0.8198 |      0.0765 | higher      | True               |
| gemma_2b_it    | 64.0000 | 3000 | Average loss            | 7.9478 |      0.2001 |     -7.7477 | lower       | True               |
| gemma_2b_it    | 64.0000 | 3000 | Classification accuracy | 0.7830 |      0.8000 |      0.0170 | higher      | True               |
| gemma_2b_it    | 64.0000 | 3000 | NLI accuracy            | 0.6100 |      0.7220 |      0.1120 | higher      | True               |
| gemma_2b_it    | 64.0000 | 3000 | PII F1                  | 0.6557 |      0.7605 |      0.1048 | higher      | True               |
| gemma_2b_it    | 64.0000 | 3000 | PII relaxed F1          | 0.7433 |      0.8336 |      0.0903 | higher      | True               |
| llama3_2_1b_it |  3.0000 | 3000 | Average loss            | 1.9121 |      0.5400 |     -1.3721 | lower       | True               |
| llama3_2_1b_it |  3.0000 | 3000 | Classification accuracy | 0.2860 |      0.6810 |      0.3950 | higher      | True               |
| llama3_2_1b_it |  3.0000 | 3000 | NLI accuracy            | 0.3640 |      0.4580 |      0.0940 | higher      | True               |
| llama3_2_1b_it |  3.0000 | 3000 | PII F1                  | 0.0000 |      0.5681 |      0.5681 | higher      | True               |
| llama3_2_1b_it |  3.0000 | 3000 | PII relaxed F1          | 0.0048 |      0.6382 |      0.6334 | higher      | True               |
| llama3_2_1b_it |  5.0000 | 3000 | Average loss            | 1.9121 |      0.5548 |     -1.3573 | lower       | True               |
| llama3_2_1b_it |  5.0000 | 3000 | Classification accuracy | 0.2860 |      0.6740 |      0.3880 | higher      | True               |
| llama3_2_1b_it |  5.0000 | 3000 | NLI accuracy            | 0.3640 |      0.4490 |      0.0850 | higher      | True               |
| llama3_2_1b_it |  5.0000 | 3000 | PII F1                  | 0.0000 |      0.5325 |      0.5325 | higher      | True               |
| llama3_2_1b_it |  5.0000 | 3000 | PII relaxed F1          | 0.0048 |      0.5948 |      0.5900 | higher      | True               |
| llama3_2_1b_it |  8.0000 | 3000 | Average loss            | 1.9121 |      0.5765 |     -1.3356 | lower       | True               |
| llama3_2_1b_it |  8.0000 | 3000 | Classification accuracy | 0.2860 |      0.7060 |      0.4200 | higher      | True               |
| llama3_2_1b_it |  8.0000 | 3000 | NLI accuracy            | 0.3640 |      0.4500 |      0.0860 | higher      | True               |
| llama3_2_1b_it |  8.0000 | 3000 | PII F1                  | 0.0000 |      0.5361 |      0.5361 | higher      | True               |
| llama3_2_1b_it |  8.0000 | 3000 | PII relaxed F1          | 0.0048 |      0.5954 |      0.5906 | higher      | True               |
| llama3_2_1b_it | 16.0000 | 3000 | Average loss            | 1.9121 |      0.6886 |     -1.2235 | lower       | True               |
| llama3_2_1b_it | 16.0000 | 3000 | Classification accuracy | 0.2860 |      0.6120 |      0.3260 | higher      | True               |
| llama3_2_1b_it | 16.0000 | 3000 | NLI accuracy            | 0.3640 |      0.4290 |      0.0650 | higher      | True               |
| llama3_2_1b_it | 16.0000 | 3000 | PII F1                  | 0.0000 |      0.4930 |      0.4930 | higher      | True               |
| llama3_2_1b_it | 16.0000 | 3000 | PII relaxed F1          | 0.0048 |      0.5525 |      0.5477 | higher      | True               |
| llama3_2_1b_it | 64.0000 | 3000 | Average loss            | 1.9121 |      0.7875 |     -1.1246 | lower       | True               |
| llama3_2_1b_it | 64.0000 | 3000 | Classification accuracy | 0.2860 |      0.6470 |      0.3610 | higher      | True               |
| llama3_2_1b_it | 64.0000 | 3000 | NLI accuracy            | 0.3640 |      0.4540 |      0.0900 | higher      | True               |
| llama3_2_1b_it | 64.0000 | 3000 | PII F1                  | 0.0000 |      0.5799 |      0.5799 | higher      | True               |
| llama3_2_1b_it | 64.0000 | 3000 | PII relaxed F1          | 0.0048 |      0.6510 |      0.6462 | higher      | True               |
| vault_gemma    |  3.0000 | 2936 | Average loss            | 3.5228 |      0.9072 |     -2.6156 | lower       | True               |
| vault_gemma    |  3.0000 | 2936 | Classification accuracy | 0.2456 |      0.2425 |     -0.0031 | higher      | False              |
| vault_gemma    |  3.0000 | 2936 | NLI accuracy            | 0.3571 |      0.3663 |      0.0092 | higher      | True               |
| vault_gemma    |  3.0000 | 2936 | PII F1                  | 0.0000 |      0.1659 |      0.1659 | higher      | True               |
| vault_gemma    |  3.0000 | 2936 | PII relaxed F1          | 0.2270 |      0.4839 |      0.2569 | higher      | True               |
| vault_gemma    |  5.0000 | 2936 | Average loss            | 3.5228 |      0.7547 |     -2.7681 | lower       | True               |
| vault_gemma    |  5.0000 | 2936 | Classification accuracy | 0.2456 |      0.2518 |      0.0062 | higher      | True               |
| vault_gemma    |  5.0000 | 2936 | NLI accuracy            | 0.3571 |      0.3571 |      0.0000 | higher      | False              |
| vault_gemma    |  5.0000 | 2936 | PII F1                  | 0.0000 |      0.1687 |      0.1687 | higher      | True               |
| vault_gemma    |  5.0000 | 2936 | PII relaxed F1          | 0.2270 |      0.4428 |      0.2158 | higher      | True               |
| vault_gemma    |  8.0000 | 2936 | Average loss            | 3.5228 |      0.7278 |     -2.7949 | lower       | True               |
| vault_gemma    |  8.0000 | 2936 | Classification accuracy | 0.2456 |      0.2467 |      0.0010 | higher      | True               |
| vault_gemma    |  8.0000 | 2936 | NLI accuracy            | 0.3571 |      0.3643 |      0.0071 | higher      | True               |
| vault_gemma    |  8.0000 | 2936 | PII F1                  | 0.0000 |      0.2439 |      0.2439 | higher      | True               |
| vault_gemma    |  8.0000 | 2936 | PII relaxed F1          | 0.2270 |      0.3835 |      0.1565 | higher      | True               |
| vault_gemma    | 16.0000 | 2936 | Average loss            | 3.5228 |      0.6357 |     -2.8870 | lower       | True               |
| vault_gemma    | 16.0000 | 2936 | Classification accuracy | 0.2456 |      0.2456 |      0.0000 | higher      | False              |
| vault_gemma    | 16.0000 | 2936 | NLI accuracy            | 0.3571 |      0.3663 |      0.0092 | higher      | True               |
| vault_gemma    | 16.0000 | 2936 | PII F1                  | 0.0000 |      0.4113 |      0.4113 | higher      | True               |
| vault_gemma    | 16.0000 | 2936 | PII relaxed F1          | 0.2270 |      0.2931 |      0.0661 | higher      | True               |
| vault_gemma    | 64.0000 | 2936 | Average loss            | 3.5228 |      0.5540 |     -2.9687 | lower       | True               |
| vault_gemma    | 64.0000 | 2936 | Classification accuracy | 0.2456 |      0.2456 |      0.0000 | higher      | False              |
| vault_gemma    | 64.0000 | 2936 | NLI accuracy            | 0.3571 |      0.3633 |      0.0061 | higher      | True               |
| vault_gemma    | 64.0000 | 2936 | PII F1                  | 0.0000 |      0.4810 |      0.4810 | higher      | True               |
| vault_gemma    | 64.0000 | 2936 | PII relaxed F1          | 0.2270 |      0.2576 |      0.0306 | higher      | True               |
