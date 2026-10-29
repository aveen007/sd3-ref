import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from other_impls import T5XXLModel, T5XXLTokenizer
from sd3_infer import T5_CONFIG, load_into


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file")
    parser.add_argument("--vocabulary_file")
    parser.add_argument("--model", default="models/t5xxl.safetensors")
    parser.add_argument(
        "--output",
        default="sd3_t5_geneval_vocab1080_compact.pt",
    )
    parser.add_argument("--extra_tokens", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def load_inputs(metadata_file, vocabulary_file):
    metadata_path = Path(metadata_file)
    with metadata_path.open(encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file if line.strip()]

    if vocabulary_file is None:
        candidate = metadata_path.with_name("object_names.txt")
        vocabulary_file = candidate if candidate.exists() else None

    vocabulary = []
    if vocabulary_file is not None:
        with Path(vocabulary_file).open(encoding="utf-8") as file:
            vocabulary = [line.strip() for line in file if line.strip()]
    return metadata, vocabulary, vocabulary_file


def choose_extra_tokens(tokenizer, count, excluded, seed):
    candidates = []
    seen = set(excluded)
    special_ids = set(tokenizer.all_special_ids)

    for piece, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        if token_id in special_ids or not piece.startswith("▁"):
            continue
        word = tokenizer.convert_tokens_to_string([piece]).strip()
        key = word.lower()
        if (
            key in seen
            or not re.fullmatch(r"[A-Za-z][A-Za-z-]{2,24}", word)
            or tokenizer.encode(word, add_special_tokens=False) != [token_id]
        ):
            continue
        seen.add(key)
        candidates.append((token_id, word))

    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} clean T5 vocabulary tokens are available")

    selected = random.Random(seed).sample(candidates, count)
    return [word for _, word in selected], [token_id for token_id, _ in selected]


def tokenize_exact(text, wrapper):
    tokenizer = wrapper.tokenizer
    normalized = text.replace("\n", " ")
    input_ids = []
    offsets = []

    for match in re.finditer(r"[^ ]+", normalized):
        encoded = tokenizer(
            match.group(),
            return_offsets_mapping=True,
        )
        word_ids = encoded["input_ids"][wrapper.tokens_start : -1]
        word_offsets = encoded["offset_mapping"][wrapper.tokens_start : -1]
        input_ids.extend(word_ids)
        offsets.extend(
            (match.start() + start, match.start() + end)
            for start, end in word_offsets
        )

    input_ids.append(wrapper.end_token)
    offsets.append((0, 0))
    content_length = len(input_ids)

    if len(input_ids) < wrapper.min_length:
        padding = wrapper.min_length - len(input_ids)
        input_ids.extend([0] * padding)
        offsets.extend([(0, 0)] * padding)

    reference_ids = [token_id for token_id, _ in wrapper.tokenize_with_weights(text)[0]]
    if input_ids != reference_ids:
        raise RuntimeError(f"Tokenizer reproduction differed for: {text!r}")

    special_tokens_mask = tokenizer.get_special_tokens_mask(
        input_ids,
        already_has_special_tokens=True,
    )
    return {
        "input_ids": input_ids,
        "offsets": offsets,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "special_tokens_mask": [bool(value) for value in special_tokens_mask],
        "content_mask": [position < content_length for position in range(len(input_ids))],
        "content_length": content_length,
    }


class NormStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values):
        if values.numel() == 0:
            return
        values = values.double()
        self.count += values.numel()
        self.total += values.sum().item()
        self.square_total += values.square().sum().item()
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())

    def export(self):
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


@torch.inference_mode()
def main(args):
    metadata, original_vocabulary, vocabulary_file = load_inputs(
        args.metadata_file,
        args.vocabulary_file,
    )
    tokenizer_wrapper = T5XXLTokenizer()
    tokenizer = tokenizer_wrapper.tokenizer

    evaluation_prompts = [item["prompt"] for item in metadata]
    excluded = {
        word.lower()
        for text in evaluation_prompts + original_vocabulary
        for word in re.findall(r"[A-Za-z][A-Za-z-]+", text)
    }
    extra_vocabulary, extra_token_ids = choose_extra_tokens(
        tokenizer,
        args.extra_tokens,
        excluded,
        args.seed,
    )

    texts = evaluation_prompts + original_vocabulary + extra_vocabulary
    splits = (
        ["geneval"] * len(evaluation_prompts)
        + ["original_vocabulary"] * len(original_vocabulary)
        + ["extra_t5_vocabulary"] * len(extra_vocabulary)
    )
    ids = (
        [f"{index:05d}" for index in range(len(evaluation_prompts))]
        + [f"original-{index:04d}" for index in range(len(original_vocabulary))]
        + [f"extra-{index:04d}" for index in range(len(extra_vocabulary))]
    )
    encoded = [tokenize_exact(text, tokenizer_wrapper) for text in texts]

    lengths = torch.tensor([len(item["input_ids"]) for item in encoded], dtype=torch.int32)
    content_lengths = torch.tensor(
        [item["content_length"] for item in encoded],
        dtype=torch.int32,
    )
    max_length = int(lengths.max())
    input_ids = torch.zeros(len(texts), max_length, dtype=torch.int32)
    model_attention_mask = torch.zeros(len(texts), max_length, dtype=torch.bool)
    content_mask = torch.zeros_like(model_attention_mask)
    special_tokens_mask = torch.ones_like(model_attention_mask)
    offset_mapping = torch.zeros(len(texts), max_length, 2, dtype=torch.int32)
    token_strings = []

    for index, item in enumerate(encoded):
        length = len(item["input_ids"])
        input_ids[index, :length] = torch.tensor(item["input_ids"], dtype=torch.int32)
        model_attention_mask[index, :length] = True
        content_mask[index, :length] = torch.tensor(item["content_mask"])
        special_tokens_mask[index, :length] = torch.tensor(item["special_tokens_mask"])
        offset_mapping[index, :length] = torch.tensor(item["offsets"], dtype=torch.int32)
        token_strings.append(item["tokens"])

    row_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), content_lengths.to(torch.int64).cumsum(0)]
    )
    hidden_states = torch.empty(
        int(row_offsets[-1]),
        T5_CONFIG["d_model"],
        dtype=torch.float16,
    )

    print("Loading native SD3 T5-XXL encoder on CPU in float32...")
    model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32).eval()
    with safe_open(args.model, framework="pt", device="cpu") as file:
        load_into(file, model.transformer, "", "cpu", torch.float32)

    groups = defaultdict(list)
    for index, length in enumerate(lengths.tolist()):
        groups[length].append(index)

    batches = []
    for length, indices in sorted(groups.items()):
        for start in range(0, len(indices), args.batch_size):
            batches.append((length, indices[start : start + args.batch_size]))

    stats = {name: NormStats() for name in ("all", "content", "padding")}
    for batch_index, (length, indices) in enumerate(batches, 1):
        batch_ids = input_ids[indices, :length].long().tolist()
        output, _ = model(batch_ids)
        stored = output.to(torch.float16).cpu()

        for local_index, prompt_index in enumerate(indices):
            start = int(row_offsets[prompt_index])
            end = int(row_offsets[prompt_index + 1])
            prompt_mask = content_mask[prompt_index, :length]
            hidden_states[start:end] = stored[local_index, prompt_mask]

        norms = stored.float().norm(dim=-1)
        batch_content_mask = content_mask[indices, :length]
        stats["all"].update(norms.flatten())
        stats["content"].update(norms[batch_content_mask])
        stats["padding"].update(norms[~batch_content_mask])

        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(batches):
            print(f"Encoded batch {batch_index}/{len(batches)}")

    active_ids = input_ids[content_mask].long()
    lookup_token_ids = torch.unique(active_ids, sorted=True)
    lookup_weight = model.transformer.get_input_embeddings().weight.detach()
    lookup_embeddings = lookup_weight[lookup_token_ids].to(torch.float16).cpu()

    artifact = {
        "format_version": 2,
        "kind": "sd3_t5_geneval_augmented_vocabulary_compact",
        "target_model": "google/t5-v1_1-xxl",
        "text_encoder_model_id": "google/t5-v1_1-xxl",
        "target_repository": "Stability-AI/sd3-ref",
        "target_representation": "final_contextual_t5_content_and_eos_positions",
        "tokenizer_model_id": "google/t5-v1_1-xxl",
        "tokenizer_vocab_size": len(tokenizer),
        "storage_dtype": "float16",
        "encoder_compute_dtype": "float32",
        "encoder_compute_device": "cpu",
        "prompt_preprocessing": "replace_newlines_with_spaces_then_tokenize_each_space_delimited_word",
        "padding_side": "right",
        "pad_token_id": 0,
        "eos_token_id": 1,
        "minimum_sequence_length": 77,
        "attention_mask_used_by_encoder": False,
        "attention_mask_semantics": "stored content and EOS positions only",
        "conditioning_layout": {
            "clip_sequence": "77 x (CLIP-L 768 + CLIP-G 1280), right-padded to 4096",
            "t5_sequence": "native T5 sequence x 4096, appended after the CLIP sequence",
            "pooled_conditioning": "native CLIP-L pooled concatenated with native CLIP-G pooled",
            "projected_component": "T5 content and EOS positions; native contextual padding is retained",
        },
        "evaluation_prompt_count": len(evaluation_prompts),
        "original_vocabulary_count": len(original_vocabulary),
        "extra_vocabulary_count": len(extra_vocabulary),
        "calculation_prompt_count": len(texts),
        "extra_vocabulary_seed": args.seed,
        "original_vocabulary": original_vocabulary,
        "extra_vocabulary": extra_vocabulary,
        "metadata_file": str(Path(args.metadata_file)),
        "vocabulary_file": str(vocabulary_file) if vocabulary_file is not None else None,
        "prompt_texts": texts,
        "conditioned_texts": texts,
        "ids": ids,
        "splits": splits,
        "chi_prompt": "",
        "input_ids": input_ids,
        "token_ids": input_ids,
        "token_strings": token_strings,
        "offset_mapping": offset_mapping,
        "special_tokens_mask": special_tokens_mask,
        "attention_mask": content_mask,
        "content_mask": content_mask,
        "model_attention_mask": model_attention_mask,
        "sequence_lengths": lengths,
        "content_lengths": content_lengths,
        "row_offsets": row_offsets,
        "Y_pred_active": hidden_states,
        "conditioning_storage": "content_and_eos_positions_only",
        "dense_shape": [len(texts), max_length, T5_CONFIG["d_model"]],
        "lookup_token_ids": lookup_token_ids.to(torch.int32),
        "lookup_embeddings": lookup_embeddings,
        "lookup_embedding_semantics": "native_T5_input_embedding_prior_without_context",
        "extra_vocabulary_token_ids": torch.tensor(extra_token_ids, dtype=torch.int32),
        "native_norms": {name: value.export() for name, value in stats.items()},
    }
    torch.save(artifact, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main(parse_args())
