#!/usr/bin/env python3
"""
Chatterbox Russian Tokenizer & Embedding Merger Script

This script:
1. Downloads the Chatterbox model from HuggingFace
2. Trains a Russian BPE tokenizer on the ToneSpeak dataset
3. Merges English and Russian tokenizers
4. Aligns embeddings using Fourier-domain merging or SLERP
5. Saves the updated model with Russian language support

Usage:
    python merge_chatterbox_ru.py \
        --model-repo ResembleAI/chatterbox \
        --dataset Vikhrmodels/ToneSpeak \
        --output-dir ./output_model \
        --vocab-size 2000 \
        --merger-type fourier
"""
import sys
sys.path.append("./chatterbox/")
from src.chatterbox.models.tokenizers.tokenizer import EnTokenizer
import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
from transliterate import translit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def setup_directories(output_dir: str) -> Path:
    """Create output directory structure."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "temp").mkdir(exist_ok=True)
    return out_path


def copy_dereference(src_dir: str, dst_dir: str) -> None:
    """Copy files from src to dst, resolving symlinks and preserving names."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    
    if dst.exists() and dst.is_file():
        dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name.startswith('.'):
            continue
        if item.is_symlink() or item.is_file():
            try:
                real_src = item.resolve()
                if real_src.is_file():
                    copied_path = shutil.copy2(real_src, dst)
                    new_path = dst / item.name
                    if new_path.exists():
                        new_path.unlink()
                    Path(copied_path).rename(new_path)
                    logger.info(f"Copied: {item.name}")
                else:
                    logger.warning(f"Skipping non-file: {item.name}")
            except Exception as e:
                logger.error(f"Failed on {item.name}: {e}")
        else:
            logger.debug(f"Skipping directory: {item.name}")


def transliterate_ru_to_latin(text: str) -> str:
    """Transliterate Russian Cyrillic to Latin using standard mapping."""
    return translit(text, 'ru', reversed=True)


def load_safetensors_to_torch(file_path: str, device: str = "cpu") -> dict:
    """Load safetensors file into a PyTorch state dict."""
    logger.info(f"Loading safetensors: {file_path}")
    state_dict = {}
    with safe_open(file_path, framework="pt", device=device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    logger.info(f"Loaded {len(state_dict)} tensors")
    return state_dict


def load_vocab_from_json(tokenizer_path: str) -> Dict[str, int]:
    """Load vocabulary from tokenizer.json."""
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state["model"]["vocab"]


# =============================================================================
# Simple Tokenizer (No Transformers Dependency)
# =============================================================================

class SimpleTokenizer:
    """Lightweight tokenizer using vocab.json directly."""
    
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.vocab = state["model"]["vocab"]
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.unk_token = "<unk>"
        self.unk_id = self.vocab.get(self.unk_token, 0)
        
    def encode(self, text: str) -> List[int]:
        """Greedy longest-match encoding."""
        ids = []
        i = 0
        sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)
        
        while i < len(text):
            matched = False
            for token in sorted_tokens:
                if text[i:i+len(token)] == token:
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(self.unk_id)
                i += 1
        return ids
    
    def decode(self, ids: List[int], clean_up_tokenization_spaces: bool = False) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(idx, self.unk_token) for idx in ids]
        text = ''.join(tokens)
        if clean_up_tokenization_spaces:
            text = text.replace(' ', '')
        return text


# =============================================================================
# Fourier Merger (Closed-Form LSTSQ + Gaussian Low-Pass Filter)
# =============================================================================

class FourierMerger(nn.Module):
    """
    Merges subtoken embeddings using Fourier Domain Downsampling.
    Closed-form least squares solution with Gaussian low-pass filtering.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8, keep_freqs: int = 4, gaussian_sigma: float = 2.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.keep_freqs = keep_freqs
        self.gaussian_sigma = gaussian_sigma
        self.num_freqs = max_seq_len // 2 + 1
        self.input_features = self.keep_freqs * 2 * dim
        
        # Gaussian low-pass filter
        freqs = torch.arange(self.num_freqs, dtype=torch.float32)
        gaussian_mask = torch.exp(-freqs**2 / (2 * gaussian_sigma**2))
        self.register_buffer('spectral_mask', gaussian_mask)
        
        # Projection parameters (solved via lstsq)
        self.register_buffer('projection_weight', torch.zeros(dim, self.input_features))
        self.register_buffer('projection_bias', torch.zeros(dim))
        self.register_buffer('is_trained', torch.tensor(False))
        
        self._training_data = {'X': [], 'Y': []}

    def _apply_filter_and_downsample(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian low-pass filter and spectral downsampling."""
        seq_fft = torch.fft.rfft(seq, dim=0)
        mask = self.spectral_mask.unsqueeze(-1).to(seq.device)
        seq_fft_filtered = seq_fft * mask
        return seq_fft_filtered[:self.keep_freqs, :]

    def _extract_features(self, seq_fft_downsampled: torch.Tensor) -> torch.Tensor:
        """Extract real/imag parts and flatten for projection."""
        real_part = seq_fft_downsampled.real
        imag_part = seq_fft_downsampled.imag
        features = torch.cat([real_part, imag_part], dim=0).flatten()
        return features

    def forward(self, subtoken_embs: List[torch.Tensor]) -> torch.Tensor:
        n = len(subtoken_embs)
        if n == 0:
            return torch.zeros(self.dim, device='cpu')
        
        seq = torch.stack(subtoken_embs)
        device = seq.device
        
        if n < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - n, self.dim, device=device)
            seq = torch.cat([seq, padding], dim=0)
        
        seq_fft_downsampled = self._apply_filter_and_downsample(seq)
        features = self._extract_features(seq_fft_downsampled)
        
        if self.is_trained.item():
            merged = self.projection_weight @ features + self.projection_bias
        else:
            merged = seq[:n].mean(dim=0)
        
        return merged

    def collect_training_pair(self, subtoken_embs: List[torch.Tensor], target_emb: torch.Tensor) -> None:
        """Collect one training pair for lstsq."""
        n = len(subtoken_embs)
        if n == 0:
            return
        
        seq = torch.stack(subtoken_embs)
        device = seq.device
        
        if n < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - n, self.dim, device=device)
            seq = torch.cat([seq, padding], dim=0)
        
        seq_fft_downsampled = self._apply_filter_and_downsample(seq)
        features = self._extract_features(seq_fft_downsampled)
        
        self._training_data['X'].append(features)
        self._training_data['Y'].append(target_emb)

    def solve_closed_form(self, reg_lambda: float = 1e-6) -> bool:
        """Solve projection via regularized least squares (ridge regression)."""
        if len(self._training_data['X']) == 0:
            logger.warning("No training data for closed-form solution")
            return False
        
        X = torch.stack(self._training_data['X'])
        Y = torch.stack(self._training_data['Y'])
        
        ones = torch.ones(X.shape[0], 1, device=X.device)
        X_aug = torch.cat([X, ones], dim=1)
        
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ Y
        
        reg_matrix = reg_lambda * torch.eye(XtX.shape[0], device=XtX.device)
        reg_matrix[-1, -1] = 0
        
        W_aug = torch.linalg.solve(XtX + reg_matrix, Xty)
        
        self.projection_weight = W_aug[:-1, :].T
        self.projection_bias = W_aug[-1, :]
        self.is_trained = torch.tensor(True)
        self._training_data = {'X': [], 'Y': []}
        
        logger.info(f"Solved projection: {self.projection_weight.shape}")
        return True

    def train_with_lstsq(self, inputs: List[List[torch.Tensor]], targets: List[torch.Tensor], 
                         reg_lambda: float = 1e-6) -> bool:
        """Train using closed-form least squares."""
        logger.info(f"Collecting {len(inputs)} training pairs...")
        
        for sub_embs, target in zip(inputs, targets):
            self.collect_training_pair(sub_embs, target)
        
        logger.info("Solving closed-form least squares...")
        success = self.solve_closed_form(reg_lambda)
        
        if success:
            self.eval()
            with torch.no_grad():
                cos_sims = []
                eval_count = min(100, len(inputs))
                for i in range(eval_count):
                    pred = self(inputs[i])
                    target = targets[i]
                    cos_sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
                    cos_sims.append(cos_sim)
                if cos_sims:
                    avg_cos = sum(cos_sims) / len(cos_sims)
                    logger.info(f"Training complete. Avg cosine similarity: {avg_cos:.4f}")
        
        return success


# =============================================================================
# SLERP Fallback
# =============================================================================

def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    return (1 - t) * v0 + t * v1


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9995) -> torch.Tensor:
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)
    norm_v0, norm_v1 = torch.norm(v0), torch.norm(v1)
    epsilon = 1e-10
    
    if norm_v0 < epsilon or norm_v1 < epsilon:
        return lerp(t, v0, v1)
    
    u0, u1 = v0 / norm_v0, v1 / norm_v1
    dot = torch.clamp(torch.sum(u0 * u1), -1.0, 1.0)
    
    if torch.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0, v1)
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    if sin_theta_0 < epsilon:
        return lerp(t, v0, v1)
    
    theta_t = theta_0 * t
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    return s0 * v0 + s1 * v1


def slerp_recursive(vectors: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if len(vectors) == 0:
        return None
    if len(vectors) == 1:
        return vectors[0]
    if len(vectors) == 2:
        return slerp(0.5, vectors[0], vectors[1])
    
    merged = []
    for i in range(0, len(vectors), 2):
        if i + 1 < len(vectors):
            merged.append(slerp(0.5, vectors[i], vectors[i+1]))
        else:
            merged.append(vectors[i])
    return slerp_recursive(merged)


# =============================================================================
# Training Data Generation
# =============================================================================

def find_training_pairs(tokenizer_old: SimpleTokenizer, embed_matrix: torch.Tensor, 
                        vocab: Dict[str, int], max_pairs: int = 500) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
    """Find subtoken → token embedding pairs for training."""
    inputs, targets = [], []
    count = 0
    max_valid_idx = embed_matrix.shape[0] - 1
    text_to_id = {token: idx for token, idx in vocab.items() if idx <= max_valid_idx}
    
    logger.info(f"Scanning vocabulary (max valid index: {max_valid_idx})...")
    
    for token_str, token_id in vocab.items():
        if count >= max_pairs:
            break
        if token_id > max_valid_idx:
            continue
        try:
            text = token_str
            ids_split = tokenizer_old.encode(text)
            if len(ids_split) == 1 and ids_split[0] == token_id:
                ids_split = tokenizer_old.encode(' ' + text)
            valid_ids = [i for i in ids_split if i <= max_valid_idx]
            
            if len(valid_ids) > 1:
                reconstructed = tokenizer_old.decode(valid_ids, clean_up_tokenization_spaces=False)
                if text in reconstructed or reconstructed.replace(' ', '') == text.replace(' ', ''):
                    sub_embs = [embed_matrix[i] for i in valid_ids]
                    target_emb = embed_matrix[token_id]
                    inputs.append(sub_embs)
                    targets.append(target_emb)
                    count += 1
        except Exception as e:
            logger.debug(f"Error processing {token_str}: {e}")
            continue
    
    # Fallback: character-level decomposition
    if count < max_pairs:
        logger.info("Trying character-level decomposition...")
        for token_str, token_id in vocab.items():
            if count >= max_pairs:
                break
            if token_id > max_valid_idx or len(token_str) < 2 or len(token_str) > 4:
                continue
            char_ids = []
            for char in token_str:
                if char in text_to_id:
                    char_id = text_to_id[char]
                    if char_id <= max_valid_idx:
                        char_ids.append(char_id)
            if len(char_ids) >= 2:
                sub_embs = [embed_matrix[i] for i in char_ids]
                targets.append(embed_matrix[token_id])
                inputs.append(sub_embs)
                count += 1
    
    logger.info(f"Found {len(inputs)} training pairs")
    return inputs, targets


def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = F.normalize(pred, p=2, dim=0)
    target_norm = F.normalize(target, p=2, dim=0)
    return 1.0 - torch.dot(pred_norm, target_norm)


def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                  mse_weight: float = 1.0, cosine_weight: float = 0.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    cosine = cosine_similarity_loss(pred, target)
    return mse_weight * mse + cosine_weight * cosine


def train_merger(merger: FourierMerger, inputs: List[List[torch.Tensor]], 
                 targets: List[torch.Tensor], reg_lambda: float = 1e-6) -> float:
    """Train FourierMerger using closed-form solution."""
    if not isinstance(merger, FourierMerger):
        logger.warning("Expected FourierMerger instance")
        return float('inf')
    
    merger.train()
    success = merger.train_with_lstsq(inputs, targets, reg_lambda)
    
    if success:
        merger.eval()
        total_loss = 0.0
        with torch.no_grad():
            for sub_embs, target in zip(inputs, targets):
                pred = merger(sub_embs)
                total_loss += combined_loss(pred, target).item()
        avg_loss = total_loss / max(len(inputs), 1)
        logger.info(f"Training complete. Final loss: {avg_loss:.6f}")
        return avg_loss
    else:
        logger.warning("Closed-form solution failed. Using fallback.")
        return float('inf')


def normalize_to_target(emb: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Normalize embedding to target norm."""
    return F.normalize(emb, p=2, dim=0) * target_norm


# =============================================================================
# Main Pipeline
# =============================================================================

def download_model(repo_id: str, output_dir: Path, hf_token: Optional[str], 
                   allow_patterns: List[str]) -> Path:
    """Download model from HuggingFace."""
    logger.info(f"Downloading {repo_id}...")
    for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath)
    return "/".join(local_path.split("/")[:-1])


def train_russian_tokenizer(dataset_name: str, vocab_size: int, min_frequency: int, 
                            output_dir: Path, num_proc: int = 4) -> str:
    """Train BPE tokenizer on Russian text from dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}'. Error: {e}")
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    def extract_text(example, name="text"):
        return {"text": example[name]}
    
    logger.info("Extracting text...")
    texts = dataset.map(extract_text, num_proc=num_proc, remove_columns=dataset.column_names)
    
    temp_dir = tempfile.mkdtemp(dir=output_dir / "temp")
    temp_file = os.path.join(temp_dir, "ru.txt")
    
    logger.info("Saving text to file...")
    with open(temp_file, "w", encoding="utf-8") as f:
        for i in range(len(texts)):
            example = texts[i]
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
            if i % 5000 == 0:
                logger.info(f"Written {i} lines")
    
    logger.info("Initializing BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        show_progress=True,
    )
    
    logger.info("Training tokenizer...")
    tokenizer.train(files=[temp_file], trainer=trainer)
    
    tokenizer_output = output_dir / "ru_tokenizer"
    tokenizer_output.mkdir(exist_ok=True)
    tokenizer.save(str(tokenizer_output / "tokenizer.json"))
    
    logger.info(f"Tokenizer saved to {tokenizer_output}")
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    shutil.rmtree(temp_dir)
    return str(tokenizer_output / "tokenizer.json")


def merge_tokenizers(old_tokenizer_path: str, new_tokenizer_path: str, 
                     output_path: str, offset: int = 704) -> None:
    """Merge two BPE tokenizers, adjusting indices."""
    logger.info("Merging tokenizers...")
    
    with open(old_tokenizer_path, "r") as f:
        state1 = json.load(f)
    with open(new_tokenizer_path, "r") as f:
        state2 = json.load(f)
    
    vocab1 = state1["model"]["vocab"]
    vocab2 = state2["model"]["vocab"]
    
    # Adjust indices in vocab2
    for key in vocab2:
        vocab2[key] += offset
    
    merges1 = state1["model"]["merges"]
    merges2 = state2["model"]["merges"]
    
    def merge_bpe_merges(m1, m2):
        seen = set()
        merged = []
        for m in m1:
            if m not in seen:
                seen.add(m)
                merged.append(m)
        for pair in m2:
            merge_str = " ".join(pair) if isinstance(pair, list) else pair
            if merge_str not in seen:
                seen.add(merge_str)
                merged.append(merge_str)
        return merged
    
    # Rebuild unified vocab
    new_vocab = {}
    next_id = 0
    for token in vocab1:
        if token not in new_vocab:
            new_vocab[token] = next_id
            next_id += 1
    for token in vocab2:
        if token not in new_vocab:
            new_vocab[token] = next_id
            next_id += 1
    
    state1["model"]["vocab"] = new_vocab
    state1["model"]["merges"] = merge_bpe_merges(merges1, merges2)
    
    with open(output_path, "w") as f:
        json.dump(state1, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Merged tokenizer saved to {output_path}")


def generate_embeddings(merger_type: str, train_merger_flag: bool, 
                        old_tokenizer_path: str, new_tokenizer_path: str,
                        model_path: str, output_path: str, device: str,
                        max_seq_len: int = 8, keep_freqs: int = 4,
                        reg_lambda: float = 1e-6) -> None:
    """Generate new embeddings for Russian tokens."""
    logger.info("Loading model...")
    chk = load_safetensors_to_torch(model_path, device)
    embed = chk["text_emb.weight"].to(device)
    embed2 = chk["text_head.weight"].to(device)
    
    dim = embed.shape[1]
    max_old_idx = embed.shape[0] - 1
    
    logger.info("Loading vocabularies...")
    vocab_old = load_vocab_from_json(old_tokenizer_path)
    vocab_new = load_vocab_from_json(new_tokenizer_path)
    logger.info(f"Old vocab: {len(vocab_old)}, New vocab: {len(vocab_new)}")
    
    tokenizer_orig = SimpleTokenizer(old_tokenizer_path)
    
    # Initialize merger
    logger.info(f"Initializing {merger_type} merger...")
    if merger_type == "fourier":
        merger = FourierMerger(dim, max_seq_len=max_seq_len, keep_freqs=keep_freqs).to(device)
    else:
        merger = None
    
    # Train merger if requested
    if merger_type == "fourier" and train_merger_flag:
        train_inputs, train_targets = find_training_pairs(
            tokenizer_orig, embed, vocab_old, max_pairs=10000
        )
        if len(train_inputs) > 50:
            logger.info("=== TRAINING FOURIER MERGER ===")
            train_merger(merger, train_inputs, train_targets, reg_lambda)
            merger.eval()
        else:
            logger.warning("Not enough training pairs. Using baseline.")
            train_merger_flag = False
    
    if merger:
        merger.eval()
    
    # Generate new embeddings
    new_vocab_size = max(vocab_new.values()) + 1
    new_embed = torch.zeros(new_vocab_size, dim, device=device)
    new_embed2 = torch.zeros(new_vocab_size, dim, device=device)
    
    # Copy shared prefix
    shared_count = min(704, embed.shape[0])
    new_embed[:shared_count] = embed[:shared_count].clone()
    new_embed2[:shared_count] = embed2[:shared_count].clone()
    
    # Precompute target norms
    orig_norms = embed.norm(dim=1)
    target_norm_mean = orig_norms.mean().item()
    target_norm_mean2 = embed2.norm(dim=1).mean().item()
    
    success_count, direct_map_count, merged_count, tot = 0, 0, 0, 0
    failed_tokens = []
    
    logger.info("Generating new embeddings for Russian tokens...")
    
    for token_id in range(shared_count, new_vocab_size):
        try:
            # Find token text
            token_text = next((t for t, idx in vocab_new.items() if idx == token_id), None)
            if token_text is None:
                raise ValueError(f"Token ID {token_id} not found")
            
            latin_text = transliterate_ru_to_latin(token_text)
            
            # Direct mapping
            if latin_text in vocab_old:
                direct_id = vocab_old[latin_text]
                if direct_id <= max_old_idx:
                    new_embed[token_id] = embed[direct_id]
                    new_embed2[token_id] = embed2[direct_id]
                    direct_map_count += 1
                    success_count += 1
                    continue
            
            # Encode and merge
            subtoken_ids = tokenizer_orig.encode(latin_text)
            valid_subtokens = [(i, embed[i]) for i in subtoken_ids if i <= max_old_idx]
            
            if not valid_subtokens:
                for char in latin_text:
                    if char in vocab_old:
                        char_id = vocab_old[char]
                        if char_id <= max_old_idx:
                            valid_subtokens.append((char_id, embed[char_id]))
            
            if not valid_subtokens:
                raise ValueError("Cannot decompose token")
            
            emb_list = [emb for _, emb in valid_subtokens]
            
            if len(emb_list) == 1:
                new_embed[token_id] = emb_list[0]
                new_embed2[token_id] = emb_list[0]
                direct_map_count += 1
            else:
                if merger_type == "slerp" or merger is None:
                    merged_emb = slerp_recursive(emb_list)
                    merged_emb2 = slerp_recursive(emb_list)
                else:
                    with torch.no_grad():
                        merged_emb = merger(emb_list)
                        merged_emb2 = merger(emb_list)
                new_embed[token_id] = merged_emb
                new_embed2[token_id] = merged_emb2
                merged_count += 1
            
            # Normalize
            new_embed[token_id] = normalize_to_target(new_embed[token_id], target_norm_mean)
            new_embed2[token_id] = normalize_to_target(new_embed2[token_id], target_norm_mean2)
            
            success_count += 1
            if success_count % 50 == 0:
                logger.info(f"Mapped {success_count} tokens (direct: {direct_map_count}, merged: {merged_count})...")
                
        except Exception as e:
            new_embed[token_id] = torch.randn(dim, device=device) * 0.02
            new_embed2[token_id] = torch.randn(dim, device=device) * 0.02
            tot += 1
            failed_tokens.append((token_id, str(e)))
    
    # Summary
    logger.info("\n=== EMBEDDING GENERATION SUMMARY ===")
    logger.info(f"Successfully mapped: {success_count} tokens")
    logger.info(f"  - Direct mapping: {direct_map_count}")
    logger.info(f"  - Merged: {merged_count}")
    logger.info(f"Failed: {tot}")
    
    if failed_tokens[:10]:
        logger.info("First 10 failures:")
        for tid, err in failed_tokens[:10]:
            token_text = next((t for t, idx in vocab_new.items() if idx == tid), "UNKNOWN")
            logger.info(f"  Token {tid} '{token_text}': {err}")
    
    # Save
    chk["text_emb.weight"] = new_embed.cpu()
    chk["text_head.weight"] = new_embed2.cpu()
    save_file(chk, output_path)
    logger.info(f"Saved updated model to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatterbox Russian Tokenizer & Embedding Merger")
    
    # Model & paths
    parser.add_argument("--model-repo", type=str, default="ResembleAI/chatterbox",
                        help="HuggingFace model repository ID")
    parser.add_argument("--dataset", type=str, default="Vikhrmodels/ToneSpeak",
                        help="Dataset for Russian text extraction")
    parser.add_argument("--output-dir", type=str, default="./output_model",
                        help="Output directory for all artifacts")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace authentication token (env var HF_TOKEN also works)")
    
    # Tokenizer training
    parser.add_argument("--vocab-size", type=int, default=200,
                        help="Vocabulary size for Russian BPE tokenizer")
    parser.add_argument("--min-frequency", type=int, default=1000,
                        help="Minimum token frequency for BPE training")
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Number of processes for dataset mapping")
    
    # Embedding merger
    parser.add_argument("--merger-type", type=str, choices=["fourier", "slerp", "none"], 
                        default="fourier", help="Embedding merging strategy")
    parser.add_argument("--train-merger", action="store_true", default=True,
                        help="Train the merger on found pairs")
    parser.add_argument("--no-train-merger", action="store_false", dest="train_merger",
                        help="Skip merger training (use fallback)")
    parser.add_argument("--max-seq-len", type=int, default=8,
                        help="Max sequence length for Fourier merger")
    parser.add_argument("--keep-freqs", type=int, default=4,
                        help="Number of frequencies to keep in Fourier merger")
    parser.add_argument("--reg-lambda", type=float, default=1e-6,
                        help="Ridge regression regularization parameter")
    
    # Advanced
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, auto-detected if None)")
    parser.add_argument("--allow-patterns", type=str, nargs="+",
                        default=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", 
                                "grapheme_mtl_merged_expanded_v1.json", "conds.pt", 
                                "Cangjie5_TC.json", "tokenizer.json"],
                        help="File patterns to download from model repo")
    parser.add_argument("--offset", type=int, default=704,
                        help="Index offset when merging tokenizers")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Auto-detect device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup
    output_dir = setup_directories(args.output_dir)
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    # Step 1: Download model
    model_cache_dir = download_model(
        repo_id=args.model_repo,
        output_dir=output_dir,
        hf_token=hf_token,
        allow_patterns=args.allow_patterns
    )
    
    # Step 2: Copy model files to working directory
    model_dir = output_dir / "model"
    copy_dereference(str(model_cache_dir), str(model_dir))
    
    # Step 3: Train Russian tokenizer
    ru_tokenizer_path = train_russian_tokenizer(
        dataset_name=args.dataset,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=output_dir,
        num_proc=args.num_proc
    )
    
    # Step 4: Merge tokenizers
    old_tokenizer = str(model_dir / "tokenizer.json")
    merged_tokenizer = str(output_dir / "en_ru_tokenizer.json")
    merge_tokenizers(old_tokenizer, ru_tokenizer_path, merged_tokenizer, offset=args.offset)
    
    # Step 5: Generate embeddings
    model_file = str(model_dir / "t3_cfg.safetensors")
    output_model = str(model_dir / "t3_cfg.safetensors")
    
    generate_embeddings(
        merger_type=args.merger_type,
        train_merger_flag=args.train_merger,
        old_tokenizer_path=old_tokenizer,
        new_tokenizer_path=merged_tokenizer,
        model_path=model_file,
        output_path=output_model,
        device=device,
        max_seq_len=args.max_seq_len,
        keep_freqs=args.keep_freqs,
        reg_lambda=args.reg_lambda
    )
    
    # Final copy
    final_tokenizer = str(model_dir / "tokenizer.json")
    shutil.copy2(merged_tokenizer, final_tokenizer)
    logger.info(f"Final model ready at: {model_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()