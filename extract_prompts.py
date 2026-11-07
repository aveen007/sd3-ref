import torch
from safetensors import safe_open
from other_impls import T5XXLModel, T5XXLTokenizer
from sd3_infer import T5_CONFIG, load_into


# Load the SD3 T5-XXL encoder weights.
model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32).eval()

with safe_open("models/t5xxl.safetensors", framework="pt", device="cpu") as f:
    load_into(f, model.transformer, "", "cpu", torch.float32)
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
        print(f"Hidden 0: {hidden[0]}", flush=True)
        
torch.save(
    {"prompt_texts": prompts, "embeddings": embeddings},
    "geneval_t5_prompt_embeddings.pt",
)
print(f"Saved {len(prompts)} prompt embeddings to geneval_t5_prompt_embeddings.pt")
