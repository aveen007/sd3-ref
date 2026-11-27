import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm.auto import tqdm

from other_impls import T5XXLModel, T5XXLTokenizer
from sd3_infer import T5_CONFIG, load_into


SEED = 20261127
VOCAB_LIMIT = 5000
TARGET_COUNT = 500
CONTEXTS_PER_TARGET = 10
SEQUENCE_LENGTH = 77
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
UNK_TOKEN_ID = 2
BOUNDARY_TOKEN_ID = 3
SPECIAL_TOKEN_IDS = {PAD_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID}

RAW_WORD_POOLS = {
    "articles": [
        "a", "an", "the", "this", "that", "these", "those", "each",
        "every", "some", "another",
    ],
    "nouns": [
        "artist", "bird", "camera", "child", "city", "cloud", "color",
        "day", "door", "evening", "field", "forest", "garden", "house",
        "image", "light", "machine", "morning", "music", "object", "ocean",
        "pattern", "person", "picture", "river", "road", "room", "scene",
        "shape", "stone", "story", "table", "tree", "water", "window",
        "world", "writer",
    ],
    "adjectives": [
        "ancient", "blue", "bright", "clear", "cold", "dark", "deep",
        "gentle", "golden", "green", "large", "modern", "new", "old",
        "quiet", "red", "sharp", "simple", "small", "soft", "strange",
        "strong", "vivid", "warm", "wide", "young",
    ],
    "verbs": [
        "appears", "becomes", "builds", "carries", "crosses", "enters",
        "finds", "follows", "frames", "holds", "leaves", "meets", "moves",
        "reflects", "rests", "sees", "shows", "stands", "surrounds", "takes",
        "travels", "watches",
    ],
    "prepositions": [
        "above", "across", "after", "around", "before", "behind", "below",
        "beside", "beyond", "by", "inside", "near", "of", "on", "outside",
        "over", "through", "toward", "under", "with", "without",
    ],
}

FIXED_TEMPLATE_WORDS = {"while", "there", "is", "and", "when", "does"}
PUNCTUATION = {".", ",", "?"}
TARGET_MARKER = object()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 5,000 native SD3 contextual T5 training embeddings."
    )
    parser.add_argument("--model", default="models/t5xxl.safetensors")
    parser.add_argument(
        "--geneval-embeddings",
        default="geneval_t5_prompt_embeddings.pt",
    )
    parser.add_argument("--output", default="t5_contextual_5000.pt")
    parser.add_argument("--prompts-output", default="contextual_prompts.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--geneval-checks", type=int, default=5)
    return parser.parse_args()


def native_word_ids(wrapper, word):
    encoded = wrapper.tokenizer(word)["input_ids"]
    return encoded[wrapper.tokens_start : -1]


def validate_word_pools(wrapper):
    pools = {}
    pool_ids = {}
    for name, words in RAW_WORD_POOLS.items():
        valid_words = []
        ids_by_word = {}
        for word in words:
            token_ids = native_word_ids(wrapper, word)
            if (
                token_ids
                and not SPECIAL_TOKEN_IDS.intersection(token_ids)
                and max(token_ids) < VOCAB_LIMIT
            ):
                valid_words.append(word)
                ids_by_word[word] = token_ids
        if not valid_words:
            raise RuntimeError(f"No valid first-5,000 words remained for {name}")
        pools[name] = valid_words
        pool_ids[name] = ids_by_word

    for word in sorted(FIXED_TEMPLATE_WORDS | PUNCTUATION):
        token_ids = native_word_ids(wrapper, word)
        if (
            not token_ids
            or SPECIAL_TOKEN_IDS.intersection(token_ids)
            or max(token_ids) >= VOCAB_LIMIT
        ):
            raise RuntimeError(
                f"Template word {word!r} is not contained in the first 5,000 IDs"
            )
    return pools, pool_ids


def collect_target_candidates(wrapper):
    tokenizer = wrapper.tokenizer
    candidates = {}
    for piece, token_id in sorted(
        tokenizer.get_vocab().items(), key=lambda item: item[1]
    ):
        if not (BOUNDARY_TOKEN_ID < token_id < VOCAB_LIMIT):
            continue
        if not piece.startswith("\u2581"):
            continue
        word = tokenizer.convert_tokens_to_string([piece]).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{1,23}", word):
            continue
        if native_word_ids(wrapper, word) != [token_id]:
            continue
        candidates[token_id] = word
    if len(candidates) < TARGET_COUNT - 1:
        raise RuntimeError(f"Only {len(candidates)} standalone target tokens found")
    return candidates


def select_target_ids(wrapper, candidates, pools, pool_ids, rng):
    # Words fixed in templates cannot be selected, avoiding accidental target collisions.
    excluded = set()
    for word in FIXED_TEMPLATE_WORDS | PUNCTUATION:
        excluded.update(native_word_ids(wrapper, word))

    selected = [BOUNDARY_TOKEN_ID]
    selected_set = {BOUNDARY_TOKEN_ID}

    # Guarantee that all requested grammatical categories occur among the targets.
    for name in ("nouns", "adjectives", "verbs", "articles", "prepositions"):
        category_ids = []
        for word in pools[name]:
            token_ids = pool_ids[name][word]
            if len(token_ids) == 1 and token_ids[0] in candidates:
                category_ids.append(token_ids[0])
        if not category_ids:
            raise RuntimeError(f"No single-token target candidates for {name}")
        for token_id in category_ids:
            if token_id not in selected_set and token_id not in excluded:
                selected.append(token_id)
                selected_set.add(token_id)

    # Fill the remaining slots round-robin across ID ranges and surface lengths.
    buckets = defaultdict(list)
    for token_id, word in candidates.items():
        if token_id in selected_set or token_id in excluded:
            continue
        id_band = min(token_id // 1000, 4)
        length_band = min((len(word) - 1) // 3, 3)
        buckets[(id_band, length_band)].append(token_id)
    for values in buckets.values():
        rng.shuffle(values)

    keys = sorted(buckets)
    while len(selected) < TARGET_COUNT:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < TARGET_COUNT:
                token_id = buckets[key].pop()
                selected.append(token_id)
                selected_set.add(token_id)
                progressed = True
        if not progressed:
            raise RuntimeError("Ran out of diverse target candidates")

    assert len(selected) == TARGET_COUNT
    assert len(set(selected)) == TARGET_COUNT
    assert not SPECIAL_TOKEN_IDS.intersection(selected)
    return selected


def choose_context_word(name, target_id, pools, pool_ids, rng):
    choices = [
        word for word in pools[name]
        if target_id not in pool_ids[name][word]
    ]
    if not choices:
        raise RuntimeError(f"No {name} context word avoids target ID {target_id}")
    return rng.choice(choices)


def context_template(index, target_id, pools, pool_ids, rng):
    def pick(name):
        return choose_context_word(name, target_id, pools, pool_ids, rng)

    templates = [
        lambda: [TARGET_MARKER, pick("verbs"), pick("articles"),
                 pick("adjectives"), pick("nouns"), "."],
        lambda: [pick("articles"), pick("adjectives"), TARGET_MARKER,
                 pick("verbs"), pick("prepositions"), pick("articles"),
                 pick("nouns"), "."],
        lambda: [pick("articles"), pick("nouns"), pick("verbs"),
                 pick("prepositions"), pick("articles"), pick("adjectives"),
                 TARGET_MARKER, "."],
        lambda: ["while", pick("articles"), pick("nouns"), pick("verbs"),
                 pick("articles"), TARGET_MARKER, pick("verbs"),
                 pick("prepositions"), pick("articles"), pick("nouns"), "."],
        lambda: ["there", "is", pick("articles"), pick("adjectives"),
                 TARGET_MARKER, pick("prepositions"), pick("articles"),
                 pick("nouns"), "."],
        lambda: [pick("prepositions"), pick("articles"), pick("nouns"), ",",
                 pick("articles"), TARGET_MARKER, pick("verbs"),
                 pick("articles"), pick("adjectives"), pick("nouns"), "."],
        lambda: [pick("articles"), pick("nouns"), "and", pick("articles"),
                 TARGET_MARKER, pick("verbs"), pick("prepositions"),
                 pick("articles"), pick("adjectives"), pick("nouns"), "."],
        lambda: ["when", pick("articles"), pick("nouns"), pick("verbs"), ",",
                 pick("articles"), pick("adjectives"), TARGET_MARKER,
                 pick("verbs"), "."],
        lambda: ["does", pick("articles"), pick("nouns"), pick("verbs"),
                 pick("articles"), TARGET_MARKER, pick("prepositions"),
                 pick("articles"), pick("nouns"), "?"],
        lambda: [pick("articles"), pick("adjectives"), pick("nouns"),
                 pick("verbs"), pick("prepositions"), pick("articles"),
                 TARGET_MARKER, "."],
    ]
    return templates[index]()


def encode_layout(words, target_word_index, target_subtoken_index, target_id, wrapper):
    tokenizer = wrapper.tokenizer
    prompt = " ".join(words)
    input_ids = []
    offsets = []
    target_position = None
    target_char_span = None
    cursor = 0

    for word_index, word in enumerate(words):
        word_start = cursor
        word_end = word_start + len(word)
        encoded = tokenizer(word, return_offsets_mapping=True)
        word_ids = encoded["input_ids"][wrapper.tokens_start : -1]
        word_offsets = encoded["offset_mapping"][wrapper.tokens_start : -1]

        if word_index == target_word_index:
            if word_ids[target_subtoken_index] != target_id:
                raise RuntimeError(
                    f"Target ID {target_id} missing from target word {word!r}"
                )
            target_position = len(input_ids) + target_subtoken_index
            target_char_span = (word_start, word_end)

        input_ids.extend(word_ids)
        offsets.extend(
            (word_start + start, word_start + end)
            for start, end in word_offsets
        )
        cursor = word_end + 1

    input_ids.append(EOS_TOKEN_ID)
    offsets.append((0, 0))
    sequence_length = len(input_ids)
    if sequence_length > SEQUENCE_LENGTH:
        raise RuntimeError(f"Prompt exceeds {SEQUENCE_LENGTH} tokens: {prompt!r}")

    input_ids.extend([PAD_TOKEN_ID] * (SEQUENCE_LENGTH - sequence_length))
    offsets.extend([(0, 0)] * (SEQUENCE_LENGTH - sequence_length))

    reference_ids = [
        token_id for token_id, _ in wrapper.tokenize_with_weights(prompt)[0]
    ]
    if input_ids != reference_ids:
        raise RuntimeError(f"Native SD3 retokenization mismatch for {prompt!r}")
    if input_ids.count(EOS_TOKEN_ID) != 1:
        raise RuntimeError(f"Expected exactly one EOS in {prompt!r}")
    target_occurrences = input_ids[:sequence_length].count(target_id)
    expected_occurrences = target_occurrences >= 1 if target_id == BOUNDARY_TOKEN_ID else target_occurrences == 1
    if not expected_occurrences:
        raise RuntimeError(
            f"Unexpected occurrence count for target ID {target_id} in {prompt!r}"
        )
    if input_ids[target_position] != target_id:
        raise RuntimeError(f"Incorrect target position for {prompt!r}")

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "offsets": offsets,
        "target_position": target_position,
        "target_char_span": target_char_span,
        "sequence_length": sequence_length,
    }


def generate_records(wrapper, seed):
    rng = random.Random(seed)
    pools, pool_ids = validate_word_pools(wrapper)
    candidates = collect_target_candidates(wrapper)
    selected_ids = select_target_ids(wrapper, candidates, pools, pool_ids, rng)
    tokenizer = wrapper.tokenizer

    boundary_ids = native_word_ids(wrapper, "a")
    if BOUNDARY_TOKEN_ID not in boundary_ids:
        raise RuntimeError("The word 'a' no longer exposes boundary token ID 3")
    boundary_subtoken_index = boundary_ids.index(BOUNDARY_TOKEN_ID)

    records = []
    for target_index, target_id in enumerate(selected_ids):
        if target_id == BOUNDARY_TOKEN_ID:
            target_surface = "a"
            target_subtoken_index = boundary_subtoken_index
        else:
            target_surface = candidates[target_id]
            target_subtoken_index = 0
        target_token = tokenizer.convert_ids_to_tokens(target_id)

        target_prompts = set()
        for context_index in range(CONTEXTS_PER_TARGET):
            words = context_template(
                context_index, target_id, pools, pool_ids, rng
            )
            target_word_index = words.index(TARGET_MARKER)
            words[target_word_index] = target_surface
            layout = encode_layout(
                words,
                target_word_index,
                target_subtoken_index,
                target_id,
                wrapper,
            )
            if layout["prompt"] in target_prompts:
                raise RuntimeError(f"Duplicate context for target ID {target_id}")
            target_prompts.add(layout["prompt"])
            records.append({
                "row_index": len(records),
                "target_index": target_index,
                "context_index": context_index,
                "target_id": target_id,
                "target_token": target_token,
                "target_surface": target_surface,
                **layout,
            })

    assert len(records) == TARGET_COUNT * CONTEXTS_PER_TARGET
    return selected_ids, records


def tensors_from_records(records):
    return {
        "input_ids": torch.tensor(
            [record["input_ids"] for record in records], dtype=torch.int32
        ),
        "target_positions": torch.tensor(
            [record["target_position"] for record in records], dtype=torch.int64
        ),
        "target_ids": torch.tensor(
            [record["target_id"] for record in records], dtype=torch.int64
        ),
        "sequence_lengths": torch.tensor(
            [record["sequence_length"] for record in records], dtype=torch.int32
        ),
        "offset_mapping": torch.tensor(
            [record["offsets"] for record in records], dtype=torch.int32
        ),
        "target_char_spans": torch.tensor(
            [record["target_char_span"] for record in records], dtype=torch.int32
        ),
    }


def validate_records(records, tensors, wrapper):
    expected_rows = TARGET_COUNT * CONTEXTS_PER_TARGET
    if len(records) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} records, got {len(records)}")
    if tensors["input_ids"].shape != (expected_rows, SEQUENCE_LENGTH):
        raise RuntimeError("Input IDs do not have shape [5000, 77]")

    counts = Counter(tensors["target_ids"].tolist())
    if len(counts) != TARGET_COUNT or set(counts.values()) != {CONTEXTS_PER_TARGET}:
        raise RuntimeError("Target IDs are not represented exactly ten times each")

    positions = tensors["target_positions"]
    rows = torch.arange(expected_rows)
    if not torch.equal(
        tensors["input_ids"][rows, positions].long(), tensors["target_ids"]
    ):
        raise RuntimeError("Stored target positions do not contain their target IDs")

    sequence_lengths = tensors["sequence_lengths"].long()
    if not torch.equal(
        tensors["input_ids"][rows, sequence_lengths - 1].long(),
        torch.full((expected_rows,), EOS_TOKEN_ID),
    ):
        raise RuntimeError("EOS is not the last active token")
    if not torch.all((tensors["input_ids"] == EOS_TOKEN_ID).sum(1) == 1):
        raise RuntimeError("Every row must contain exactly one EOS")

    positions_grid = torch.arange(SEQUENCE_LENGTH)[None, :]
    padding_mask = positions_grid >= sequence_lengths[:, None]
    if not torch.all(tensors["input_ids"][padding_mask] == PAD_TOKEN_ID):
        raise RuntimeError("Right-padding contains non-PAD IDs")
    active_mask = ~padding_mask
    active_ids = tensors["input_ids"][active_mask]
    if int(active_ids.max()) >= VOCAB_LIMIT:
        raise RuntimeError("A prompt uses a token outside the first 5,000 IDs")

    prompts_by_target = defaultdict(set)
    for record in records:
        prompts_by_target[record["target_id"]].add(record["prompt"])
        retokenized = [
            token_id
            for token_id, _ in wrapper.tokenize_with_weights(record["prompt"])[0]
        ]
        if retokenized != record["input_ids"]:
            raise RuntimeError(
                f"Retokenization failed for row {record['row_index']}"
            )
    if any(len(prompts) != CONTEXTS_PER_TARGET for prompts in prompts_by_target.values()):
        raise RuntimeError("Contexts are not distinct within every target")


@torch.inference_mode()
def validate_geneval_compatibility(model, wrapper, path, check_count):
    ground_truth = torch.load(path, map_location="cpu", weights_only=False)
    prompts = ground_truth["prompt_texts"]
    references = ground_truth["embeddings"]
    count = min(check_count, len(prompts))
    if count < 1:
        raise RuntimeError("GenEval compatibility check needs at least one prompt")
    if count == 1:
        indices = [0]
    else:
        indices = sorted({
            round(index * (len(prompts) - 1) / (count - 1))
            for index in range(count)
        })

    scores = []
    minimum = 1.0
    for index in indices:
        hidden, _ = model.encode_token_weights(
            wrapper.tokenize_with_weights(prompts[index])
        )
        reference = references[index].float()
        if hidden[0].shape != reference.shape:
            raise RuntimeError(f"GenEval shape mismatch at prompt {index}")
        cosine = F.cosine_similarity(hidden[0].float(), reference, dim=-1)
        scores.append(cosine.mean())
        minimum = min(minimum, float(cosine.min()))

    mean_cosine = float(torch.stack(scores).mean())
    print(
        "GenEval compatibility cosine: "
        f"mean={mean_cosine:.8f}, min={minimum:.8f}, prompts={len(indices)}"
    )
    if mean_cosine < 0.9999 or minimum < 0.999:
        raise RuntimeError("GenEval compatibility validation failed")
    return ground_truth, {
        "checked_prompt_indices": indices,
        "mean_cosine": mean_cosine,
        "minimum_cosine": minimum,
    }


@torch.inference_mode()
def extract_embeddings(model, wrapper, records, tensors):
    embeddings = torch.empty(
        len(records), T5_CONFIG["d_model"], dtype=torch.float16
    )
    for index, record in enumerate(tqdm(records, desc="Encoding contextual T5")):
        token_weight_pairs = wrapper.tokenize_with_weights(record["prompt"])
        current_ids = [token_id for token_id, _ in token_weight_pairs[0]]
        if current_ids != tensors["input_ids"][index].tolist():
            raise RuntimeError(f"Extraction retokenization mismatch at row {index}")
        hidden, _ = model.encode_token_weights(token_weight_pairs)
        embeddings[index] = hidden[0, record["target_position"]].half()
    return embeddings


def contextual_cosine_statistics(embeddings):
    grouped = embeddings.float().reshape(
        TARGET_COUNT, CONTEXTS_PER_TARGET, T5_CONFIG["d_model"]
    )
    normalized = F.normalize(grouped, dim=-1)
    similarities = normalized @ normalized.transpose(1, 2)
    pair_mask = torch.triu(
        torch.ones(CONTEXTS_PER_TARGET, CONTEXTS_PER_TARGET, dtype=torch.bool),
        diagonal=1,
    )
    per_target = similarities[:, pair_mask].mean(1)
    return float(per_target.mean()), per_target.cpu()


def json_record(record, within_token_cosine):
    return {
        "row_index": record["row_index"],
        "target_index": record["target_index"],
        "context_index": record["context_index"],
        "prompt": record["prompt"],
        "target_token": record["target_token"],
        "target_surface": record["target_surface"],
        "target_id": record["target_id"],
        "target_position": record["target_position"],
        "target_char_span": list(record["target_char_span"]),
        "sequence_length": record["sequence_length"],
        "within_token_cosine": within_token_cosine,
    }


@torch.inference_mode()
def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer_wrapper = T5XXLTokenizer()
    selected_ids, records = generate_records(tokenizer_wrapper, args.seed)
    tensors = tensors_from_records(records)
    validate_records(records, tensors, tokenizer_wrapper)

    print("Loading native SD3 T5-XXL encoder on CPU in float32...")
    model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32).eval()
    with safe_open(args.model, framework="pt", device="cpu") as file:
        load_into(file, model.transformer, "", "cpu", torch.float32)

    ground_truth, geneval_metrics = validate_geneval_compatibility(
        model,
        tokenizer_wrapper,
        args.geneval_embeddings,
        args.geneval_checks,
    )
    overlap = set(record["prompt"] for record in records).intersection(
        ground_truth["prompt_texts"]
    )
    if overlap:
        raise RuntimeError("Generated prompts overlap with GenEval prompts")
    del ground_truth

    embeddings = extract_embeddings(model, tokenizer_wrapper, records, tensors)
    if embeddings.shape != (TARGET_COUNT * CONTEXTS_PER_TARGET, 4096):
        raise RuntimeError("Embedding tensor does not have shape [5000, 4096]")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("NaN or Inf found in contextual embeddings")

    overall_cosine, per_target_cosine = contextual_cosine_statistics(embeddings)
    selected_target_tokens = [
        tokenizer_wrapper.tokenizer.convert_ids_to_tokens(token_id)
        for token_id in selected_ids
    ]

    artifact = {
        "format_version": 1,
        "kind": "native_sd3_t5_contextual_500x10",
        "embeddings": embeddings.contiguous(),
        "prompt_texts": [record["prompt"] for record in records],
        **tensors,
        "selected_target_ids": torch.tensor(selected_ids, dtype=torch.int64),
        "selected_target_tokens": selected_target_tokens,
        "target_count": TARGET_COUNT,
        "contexts_per_target": CONTEXTS_PER_TARGET,
        "seed": args.seed,
        "target_model": "google/t5-v1_1-xxl",
        "target_repository": "Stability-AI/sd3-ref",
        "encoder_compute_dtype": "float32",
        "storage_dtype": "float16",
        "sequence_length": SEQUENCE_LENGTH,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "attention_mask_used_by_encoder": False,
        "within_token_cosine_overall": overall_cosine,
        "within_token_cosine_per_target": per_target_cosine,
        "geneval_compatibility": geneval_metrics,
    }
    torch.save(artifact, args.output)

    per_target_json = [
        {
            "target_index": index,
            "target_id": token_id,
            "target_token": selected_target_tokens[index],
            "within_token_cosine": float(per_target_cosine[index]),
        }
        for index, token_id in enumerate(selected_ids)
    ]
    json_payload = {
        "seed": args.seed,
        "target_count": TARGET_COUNT,
        "contexts_per_target": CONTEXTS_PER_TARGET,
        "within_token_cosine_overall": overall_cosine,
        "within_token_cosine_per_target": per_target_json,
        "records": [
            json_record(
                record,
                float(per_target_cosine[record["target_index"]]),
            )
            for record in records
        ],
    }
    Path(args.prompts_output).write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Within-token cosine overall: {overall_cosine:.8f}")
    for item in per_target_json:
        print(
            f"target_id={item['target_id']:4d} "
            f"token={item['target_token']!r} "
            f"within_token_cosine={item['within_token_cosine']:.8f}"
        )
    print(f"Saved {args.output}")
    print(f"Saved {args.prompts_output}")


if __name__ == "__main__":
    main(parse_args())
