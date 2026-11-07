import torch
from safetensors import safe_open
from other_impls import T5XXLModel, T5XXLTokenizer
from sd3_infer import T5_CONFIG, load_into


# Load the SD3 T5-XXL encoder weights.
model = T5XXLModel(T5_CONFIG, device="cuda:0", dtype=torch.float16).eval()
model.transformer.encoder.embed_tokens.to(dtype=torch.float16)
model.transformer.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.to(
    dtype=torch.float16
)
with safe_open("models/t5xxl.safetensors", framework="pt", device="cpu") as f:
    load_into(f, model.transformer, "", "cuda:0", torch.float16)

# Use the prompts saved in the GenEval conditioning artifact.
data = torch.load(
    "sd3_llava_geneval_vocab1080_projected_compact_token_scale.pt",
    map_location="cpu",
    weights_only=False,
)
prompts = data["prompt_texts"]
tokenizer = T5XXLTokenizer()

# Encode each complete prompt with T5 and save its token-level hidden states.
embeddings = []
with torch.inference_mode():
    for prompt in prompts:
        hidden, _ = model.encode_token_weights(tokenizer.tokenize_with_weights(prompt))
        embeddings.append(hidden[0].half().cpu())
        # print(f"Encoded prompt: {prompt}", flush=True)
        # print(f"Hidden shape: {hidden[0].shape}", flush=True)
        # print(f"Hidden dtype: {hidden[0].dtype}", flush=True)
        # print(f"Hidden 0: {hidden[0]}", flush=True)
        # print(f"Hidden 1: {hidden[0].half}", flush=True)
        pairs = tokenizer.tokenize_with_weights(prompts[0])[0]
        device = model.transformer.encoder.embed_tokens.weight.device
        ids = torch.tensor([[token_id for token_id, _ in pairs]], device=device)

        print("Prompt:", repr(prompts[0]))
        print("First token IDs:", ids[0, :25].tolist())
        print("Non-padding IDs:", int((ids != 0).sum()))

        x = model.transformer.encoder.embed_tokens(ids)
        print("After embedding:", x.float().norm().item(), x.float().abs().max().item())

        past_bias = None
        for i, block in enumerate(model.transformer.encoder.block):
            x, past_bias = block(x, past_bias)
            if i in (0, 1, 23):
                print(f"After block {i}:", x.float().norm().item(),
                    x.float().abs().max().item())

        x = model.transformer.encoder.final_layer_norm(x)
        print("After final norm:", x.float().norm().item(),
            x.float().abs().max().item())
torch.save(
    {"prompt_texts": prompts, "embeddings": embeddings},
    "geneval_t5_prompt_embeddings.pt",
)
print(f"Saved {len(prompts)} prompt embeddings to geneval_t5_prompt_embeddings.pt")
