
import torch
import gc
from safetensors import safe_open
from tqdm.auto import tqdm
from other_impls import T5XXLModel
from sd3_infer import T5_CONFIG, load_into

assert torch.cuda.device_count() >= 2

# Load in full precision on CPU
model = T5XXLModel(
    T5_CONFIG,
    device="cpu",
    dtype=torch.float32
).eval()

with safe_open("models/t5xxl.safetensors", framework="pt", device="cpu") as f:
    load_into(f, model.transformer, "", "cpu", torch.float32)

# Split encoder across GPUs
stack = model.transformer.encoder
split = len(stack.block) // 2

stack.embed_tokens.to("cuda:0")

for i, block in enumerate(stack.block):
    block.to("cuda:0" if i < split else "cuda:1")

stack.final_layer_norm.to("cuda:1")

# Transfer hidden states and attention bias at GPU boundary
def move_to_gpu1(module, inputs):
    x, bias = inputs
    x = x.to("cuda:1")
    if bias is not None:
        bias = bias.to("cuda:1")
    return (x, bias)

hook = stack.block[split].register_forward_pre_hook(move_to_gpu1)

gc.collect()
torch.cuda.empty_cache()

for i in range(2):
    free, total = torch.cuda.mem_get_info(i)
    print(f"GPU {i}: {free / 1024**3:.2f} GiB free")

# Generate embeddings
ids = torch.tensor([i for _, i in proj_vocab])
parts = []

with torch.inference_mode():
    for batch in tqdm(ids.split(64)):
        h, _ = model(batch[:, None].tolist())
        parts.append(h[:, 0].float().cpu())

embeds = torch.cat(parts, dim=0)

assert torch.isfinite(embeds).all(), "NaN or Inf in embeddings"
assert torch.count_nonzero(embeds).item() > 0

torch.save(embeds, "t5_proj_vocab_embeds.pt")

print("Shape:", embeds.shape)
print("Dtype:", embeds.dtype)
print("Maximum:", embeds.abs().max().item())
