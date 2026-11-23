import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from other_impls import T5XXLModel
from sd3_infer import T5_CONFIG, load_into


EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0
SEQUENCE_LENGTH = 77


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract native SD3 T5-XXL embeddings for single/repeated tokens."
    )
    parser.add_argument("--model", default="models/t5xxl.safetensors")
    parser.add_argument("--output", default="t5_augmented_embeddings.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def main(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"T5 weights not found: {args.model}")

    token_ids = torch.arange(4, 260, dtype=torch.long)
    aug_rows = token_ids.repeat_interleave(2)

    # For each token: [token, EOS, PAD, ...], then
    # [token, token, EOS, PAD, ...]. As in the SD3 implementation, no
    # attention mask is passed, so all 77 positions participate in attention.
    input_ids = torch.full(
        (len(aug_rows), SEQUENCE_LENGTH),
        PAD_TOKEN_ID,
        dtype=torch.long,
    )
    input_ids[0::2, 0] = token_ids
    input_ids[0::2, 1] = EOS_TOKEN_ID
    input_ids[1::2, 0] = token_ids
    input_ids[1::2, 1] = token_ids
    input_ids[1::2, 2] = EOS_TOKEN_ID

    extract_positions = torch.tensor([0, 1], dtype=torch.long).repeat(len(token_ids))
    Y_aug = torch.empty(len(aug_rows), T5_CONFIG["d_model"], dtype=torch.float16)

    print("Loading native SD3 T5-XXL encoder on CPU in float32...")
    model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32).eval()
    with safe_open(args.model, framework="pt", device="cpu") as file:
        load_into(file, model.transformer, "", "cpu", torch.float32)

    for start in range(0, len(input_ids), args.batch_size):
        end = min(start + args.batch_size, len(input_ids))
        output, _ = model(input_ids[start:end].tolist())
        batch_rows = torch.arange(end - start)
        Y_aug[start:end] = output[
            batch_rows,
            extract_positions[start:end],
        ].to(torch.float16).cpu()
        print(f"Encoded {end}/{len(input_ids)}", flush=True)

    assert Y_aug.shape == (512, 4096)
    assert aug_rows.shape == (512,)
    torch.save(
        {
            "Y_aug": Y_aug.contiguous(),
            "aug_rows": aug_rows.contiguous(),
        },
        args.output,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main(parse_args())
