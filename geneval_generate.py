import argparse
import json
import os

import torch

from sd3_infer import SD3Inferencer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file")
    parser.add_argument("--outdir", default="geneval_sd3_medium")
    parser.add_argument("--model", default="models/sd3_medium.safetensors")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--projected_conditioning")
    parser.add_argument("--tag")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def replace_t5_active(conditioning, projected, row_offsets, prompt_index):
    cross_attention, pooled = conditioning
    active_mask = projected["attention_mask"][prompt_index].bool()
    start = int(row_offsets[prompt_index])
    end = int(row_offsets[prompt_index + 1])
    t5_sequence = cross_attention[:, 77 : 77 + active_mask.shape[0]]
    t5_sequence[0, active_mask] = projected["Y_pred_active"][start:end].to(
        t5_sequence.dtype
    )
    return cross_attention, pooled


@torch.no_grad()
def main(args):
    with open(args.metadata_file, encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file]

    end = len(metadata) if args.end is None else min(args.end, len(metadata))
    # for index in range(args.start, end):
    selected_indices = [
        index
        for index in range(args.start, end)
        if args.tag is None or metadata[index].get("tag") == args.tag
    ]
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]
    print(f"Selected {len(selected_indices)} prompts", flush=True)
    if not selected_indices:
        return

    projected = None
    projected_offsets = None
    if args.projected_conditioning is not None:
        projected = torch.load(
            args.projected_conditioning,
            map_location="cpu",
            weights_only=False,
        )
        projected_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long),
            projected["attention_mask"].sum(1).cumsum(0),
        ])

    inferencer = SD3Inferencer()
    inferencer.load(args.model, None, args.shift)
    negative_conditioning = inferencer.get_cond("")

    for selected_index, index in enumerate(selected_indices, 1):
        item = metadata[index]
        prompt_dir = os.path.join(args.outdir, f"{index:05d}")
        sample_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)

        with open(os.path.join(prompt_dir, "metadata.jsonl"), "w", encoding="utf-8") as file:
            json.dump(item, file)

        missing = [
            sample_index
            for sample_index in range(args.n_samples)
            if not os.path.exists(os.path.join(sample_dir, f"{sample_index:05d}.png"))
        ]
        if not missing:
            continue

        print(
            f"Prompt {selected_index}/{len(selected_indices)} "
            f"[{item.get('tag')}]: {item['prompt']}",
            flush=True,
        )
        # conditioning = inferencer.get_cond(item["prompt"])
        conditioning = inferencer.get_cond(item["prompt"])
        if projected is not None:
            conditioning = replace_t5_active(
                conditioning,
                projected,
                projected_offsets,
                index,
            )
        for sample_index in missing:
            latent = inferencer.get_empty_latent(args.width, args.height)
            sampled_latent = inferencer.do_sampling(
                latent,
                args.seed + sample_index,
                conditioning,
                negative_conditioning,
                args.steps,
                args.cfg_scale,
            )
            image = inferencer.vae_decode(sampled_latent)
            output_path = os.path.join(sample_dir, f"{sample_index:05d}.png")
            temporary_path = output_path + ".tmp.png"
            image.save(temporary_path)
            os.replace(temporary_path, output_path)


if __name__ == "__main__":
    main(parse_args())
