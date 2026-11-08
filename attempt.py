import gc
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from safetensors import safe_open
from other_impls import T5XXLModel
from sd3_infer import T5_CONFIG, load_into

# Keep the complete model and all saved embeddings in float32.
tok = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
vocab = sorted(tok.get_vocab().items(), key=lambda item: item[1])[:5000]
ids = torch.tensor([token_id for _, token_id in vocab], dtype=torch.long)

model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32).eval()
with safe_open("models/t5xxl.safetensors", framework="pt", device="cpu") as f:
    load_into(f, model.transformer, "", "cpu", torch.float32)
stack = model.transformer.encoder
assert len(stack.block) == 24, "Expected 24 SD3 T5-XXL encoder blocks"

# Run six blocks per stage: 0-5 on GPU 0, 6-11 on GPU 1,
# 12-17 on GPU 0, 18-23 on GPU 1. Offload each completed
# stage back to CPU before loading the next one.
BATCH_SIZE = 64
STAGE_SIZE = 6
VRAM_RESERVE_GIB = 2.5

def weight_bytes(modules):
    return sum(p.numel() * p.element_size()
               for m in modules for p in m.parameters())

with torch.inference_mode():
    batches = ids.split(BATCH_SIZE)
    hidden = [stack.embed_tokens(batch[:, None]).cpu() for batch in batches]
    biases = [None] * len(hidden)

    for start in range(0, len(stack.block), STAGE_SIZE):
        end = min(start + STAGE_SIZE, len(stack.block))
        blocks = list(stack.block[start:end])
        gpu_index = (start // STAGE_SIZE) % 2
        dev = torch.device(f"cuda:{gpu_index}")

        torch.cuda.empty_cache()
        needed = weight_bytes(blocks)
        free, _ = torch.cuda.mem_get_info(gpu_index)
        reserve = int(VRAM_RESERVE_GIB * 1024**3)
        if free < needed + reserve:
            print(f"Stage {start}:{end}: insufficient free GPU {gpu_index} memory; using CPU")
            dev = torch.device("cpu")
        else:
            print(f"Stage {start}:{end}: {dev}; weights {needed/1024**3:.2f} GiB; "
                  f"free {free/1024**3:.2f} GiB")

        try:
            for block in blocks:
                block.to(dev)
        except torch.cuda.OutOfMemoryError:
            print("GPU transfer ran out of memory; retrying stage on CPU")
            for block in blocks:
                block.cpu()
            torch.cuda.empty_cache()
            dev = torch.device("cpu")

        def run_stage(where):
            outputs, out_biases = [], []
            with torch.inference_mode():
                for h_cpu, bias_cpu in tqdm(zip(hidden, biases), total=len(hidden),
                                            desc=f"Blocks {start}-{end-1} ({where})"):
                    h = h_cpu.to(where)
                    bias = bias_cpu.to(where) if bias_cpu is not None else None
                    for block in blocks:
                        h, bias = block(h, bias)
                    outputs.append(h.cpu())
                    out_biases.append(bias.cpu() if bias is not None else None)
            return outputs, out_biases

        try:
            next_hidden, next_biases = run_stage(dev)
        except torch.cuda.OutOfMemoryError:
            print("GPU stage ran out of memory; retrying stage on CPU")
            for block in blocks:
                block.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            next_hidden, next_biases = run_stage(torch.device("cpu"))

        for block in blocks:
            block.cpu()
        del blocks, hidden, biases
        hidden, biases = next_hidden, next_biases
        del next_hidden, next_biases
        gc.collect()
        torch.cuda.empty_cache()

    # Matches T5Stack.forward's final layer normalization.
    embeds = torch.cat([stack.final_layer_norm(h)[:, 0] for h in hidden], dim=0)

assert embeds.dtype == torch.float32
assert embeds.shape == (len(vocab), T5_CONFIG["d_model"])
assert torch.isfinite(embeds).all(), "NaN/Inf detected"
assert torch.count_nonzero(embeds).item() > 0, "All-zero embeddings detected"

torch.save(embeds, "t5_proj_vocab_embeds.pt")
print("Saved", embeds.shape, embeds.dtype, "to t5_proj_vocab_embeds.pt")