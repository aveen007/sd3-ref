from transformers import AutoTokenizer
import torch
from safetensors import safe_open
from other_impls import T5XXLModel
from sd3_infer import T5_CONFIG, load_into

# Load the vocabulary from the T5-XXL tokenizer
tok = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
vocab = sorted(tok.get_vocab().items(), key=lambda x: x[1])  # (token, ID)
print(len(vocab), vocab[:100])
proj_vocab=vocab[:15000]


# load model weights
model = T5XXLModel(T5_CONFIG, device="cuda:0", dtype=torch.float16).eval()
model.transformer.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.to(
    dtype=torch.float16
)
with safe_open("models/t5xxl.safetensors", framework="pt", device="cpu") as f:
    load_into(f, model.transformer, "", "cuda:0", torch.float16)


# split for gpu usage

ids = torch.tensor([i for _, i in proj_vocab])
parts = []
with torch.inference_mode():
    for batch in ids.split(8):
        h, _ = model(batch[:, None].tolist())
        parts.append(h[:, 0].half().cpu())

# concat and save
embeds = torch.cat(parts)
torch.save(embeds, "t5_proj_vocab_embeds.pt")
print(embeds.shape)

