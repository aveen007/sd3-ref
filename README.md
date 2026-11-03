# SD3 with projected LLaVA conditioning

This fork projects LLaVA text features into SD3 Medium's T5 conditioning space for GenEval. For the original setup and model instructions, see [README (base repo).md](README%20%28base%20repo%29.md).

## Code map

| Stage | File |
|---|---|
| Extract native SD3/T5 targets | [`extract_t5_embeddings.py`](extract_t5_embeddings.py) |
| Extract LLaVA features and fit the projection on Kaggle | [`sd3-book.ipynb`](sd3-book.ipynb) |
| Clean tracked copy of the Kaggle notebook | [`sd3-vlm-embeddings-for-image-generation.ipynb`](sd3-vlm-embeddings-for-image-generation.ipynb) |
| Load and inject projected conditioning | [`geneval_generate.py`](geneval_generate.py) |
| Recorded evaluation results | [`GENEVAL_BASELINE.md`](GENEVAL_BASELINE.md) |

## Run

Install dependencies and place these weights in `models/`:

- `clip_g.safetensors`
- `clip_l.safetensors`
- `t5xxl.safetensors`
- `sd3_medium.safetensors`

Get GenEval's `evaluation_metadata.jsonl` and `object_names.txt`, then extract the native T5 targets:

```bash
python extract_t5_embeddings.py path/to/evaluation_metadata.jsonl \
  --vocabulary_file path/to/object_names.txt \
  --extra_tokens 1080 \
  --output sd3_t5_geneval_vocab1080_compact.pt
```

Upload `sd3_t5_geneval_vocab1080_compact.pt` to Kaggle, update `SD3_TARGET_PATH` in `sd3-book.ipynb`, and run the notebook. It downloads `llava-hf/llava-1.5-7b-hf`, extracts the LLaVA embeddings, fits the projection, and writes:

- `sd3_llava_geneval_vocab1080_projector_token_scale.pt`
- `sd3_llava_geneval_vocab1080_projected_compact_token_scale.pt`

Download the projected compact file and place it in the repository root. The evaluated file is already present there.

Generate the matched native and projected GenEval images:

```bash
python geneval_generate.py path/to/evaluation_metadata.jsonl \
  --outdir geneval_sd3_native

python geneval_generate.py path/to/evaluation_metadata.jsonl \
  --outdir geneval_sd3_projected \
  --projected_conditioning sd3_llava_geneval_vocab1080_projected_compact_token_scale.pt
```

`geneval_generate.py` loads the projected file through `--projected_conditioning` and replaces the active T5 content/EOS states while retaining native CLIP conditioning and contextual T5 padding. Score both output directories with [GenEval](https://github.com/djghosh13/geneval).

## Results

Matched settings: 553 prompts, four images per prompt, 1024x1024, 28 steps, CFG 5, shift 3, seeds 0-3.

| Metric | Native SD3 Medium | Projected T5 |
|---|---:|---:|
| Correct images | 70.52% | 66.50% |
| Correct prompts | 86.62% | 84.81% |
| Overall GenEval | **0.71390** | **0.67814** |

See [`GENEVAL_BASELINE.md`](GENEVAL_BASELINE.md) for the complete category breakdown.
