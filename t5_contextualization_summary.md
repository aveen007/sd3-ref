# T5-to-T5 Contextualization — Experiment Summary

**Goal:** Approximate contextual T5-XXL token embeddings from isolated T5-XXL embeddings using **PCA + closed-form ridge regression**, without fine-tuning.

**Setup**
- **Inputs:** Precomputed isolated T5 embeddings for the first 5,000 token IDs (4,096 dimensions).
- **Fitting data:** 3,200 independent synthetic sentences; 34,432 eligible token positions.
- **Evaluation:** 553 held-out GenEval prompts; 3,897 eligible token positions.
- **Consistency fix:** Reproduced SD3's per-word tokenization and 77-token, unmasked-padding encoding. The resulting Hugging Face T5 output matched the original GenEval cache at **0.99998 cosine** on a checked prompt.

**Method:** For each token, compress a **9-token window** (radius 4) with PCA (32/64/128 dimensions per token). Fit ridge regression (λ = 0.1) to predict the **contextual-minus-isolated residual**, then add the isolated embedding back. No gradient-based training.

| Evaluation on GenEval | Mean cosine | Relative L2 |
|---|---:|---:|
| Isolated T5 baseline | 0.3609 | — |
| PCA 32 + ridge | 0.4669 | 0.9323 |
| **PCA 64 + ridge** | **0.4702** | **0.9327** |
| PCA 128 + ridge | 0.4698 | 0.9332 |

**Finding:** The mapping improves over isolated T5 but generalizes poorly from independent synthetic sentences to GenEval. An earlier **0.8442** cosine was obtained with a GenEval-only train/test split, which shared prompt patterns and used a different token-alignment procedure; it is **not directly comparable** to this evaluation.

**Next:** Check fitting-set performance, a contextual-mean baseline, and sentence-length/distribution effects before changing the method.
