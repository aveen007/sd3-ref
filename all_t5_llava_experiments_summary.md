# LLaVA → T5 / SD3: Experiment Summary

**Goal:** Replace T5-XXL text representations using aligned LLaVA embeddings and lightweight, closed-form mappings, without fine-tuning either encoder.

## Setup
- **Source:** LLaVA-1.5-7B input-token embeddings (4,096-D), aligned to T5 token positions using SEAM / Reverse-SEAM.
- **Target:** Frozen T5-v1.1-XXL *final encoder outputs* (4,096-D), not its raw embedding table.
- **Vocabulary:** First 5,000 T5 token IDs; **evaluation:** 553 GenEval prompts.

## Experiments and results

| Experiment | Main finding / result |
|---|---|
| **Tokenizer alignment** (SEAM / Reverse-SEAM) | Fixed word-boundary collisions (e.g., `▁we` vs. `we`) that mapped distinct T5 tokens to the same source vector. |
| **LLaVA → T5 KRR** (poly degree 4, coef0 1.3, λ=0.001; normalized/Cholesky-whitened spaces) | Nearly perfect fit on 5,000 isolated training tokens (~**0.9998** cosine), but poor match to *contextual* GenEval T5 outputs (**0.1392** cosine whitened; **0.3892** cosine after reconstruction). |
| **KRR on held-out isolated tokens** (4,000 fit / 1,000 test) | **0.3913** whitened cosine and **82.6%** top-1 retrieval among held-out tokens: isolated-token alignment generalizes considerably better than contextual-output prediction. |
| **Contextualization baselines** (442 GenEval prompts fit / 111 test) | Isolated T5 **0.3663**; contextual mean **0.5833**; affine **0.5858**; adaptive local mixture **0.5859**; Gaussian mixture ~**0.5858** cosine. |
| **Per-dimension residual ridge** (9-token window, radius 4) | Token-only **0.6119**; nine-token window **0.6802** cosine. |
| **Cross-dimensional PCA + ridge** (9-token window, λ=0.1) | PCA 32: **0.8185**; PCA 64: **0.8412**; PCA 128: **~0.8442** cosine on held-out GenEval prompts. |
| **Independent-prompt transfer** (3,200 new synthetic sentences fit → 553 GenEval test) | Isolated baseline **0.3609**; PCA 32: **0.4669**; PCA 64: **0.4702**; PCA 128: **0.4698**. Poor cross-distribution transfer. |

## Important diagnostics
- The **isolated T5 cache** agrees with Hugging Face T5 on a single-token test (**0.9999995** cosine).
- Initial contextual-cache mismatch (~**0.43–0.51** cosine) arose from differing **SD3 preprocessing**, not evidence of a different encoder: SD3 tokenizes words separately, pads to **77** tokens, and lets padding participate in attention. Reproducing this gave **0.99998** cosine on a checked GenEval prompt.
- The independent corpus was therefore re-encoded using SD3 conventions (**3,200 sentences, 47,482 active tokens**). Ridge used **34,432** eligible training positions; GenEval evaluation used **3,897**.
- The **~0.8442** and **0.4702** figures are **not strictly controlled comparisons**: the earlier split used GenEval-like sentences and an earlier token-alignment procedure; the latter uses independent synthetic sentences and SD3-consistent tokenization.

**Conclusion:** Isolated LLaVA→T5 alignment is substantially more successful than direct contextual prediction. PCA + local ridge reconstructs contextual T5 well *within* GenEval-like prompts, but its transfer to independent sentence distributions is weak. No neural network fine-tuning was performed; KRR and ridge coefficients were fitted in closed form.
