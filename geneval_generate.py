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
    return parser.parse_args()


@torch.no_grad()
def main(args):
    with open(args.metadata_file, encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file]

    end = len(metadata) if args.end is None else min(args.end, len(metadata))
    inferencer = SD3Inferencer()
    inferencer.load(args.model, None, args.shift)
    negative_conditioning = inferencer.get_cond("")

    for index in range(args.start, end):
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

        print(f"Prompt {index + 1}/{end}: {item['prompt']}")
        conditioning = inferencer.get_cond(item["prompt"])
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
