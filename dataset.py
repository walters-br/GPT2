"""
EECE 4520 - Milestone 3 Part 1
Dataset utilities: load Wikitext-2, encode with the BPE tokenizer,
and build sliding-window PyTorch Datasets + DataLoaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from tokenizer_singleton import TokenizerSingleton


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset — fixed-length sliding-window chunks
# ─────────────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Wraps a flat list of token IDs into overlapping (input, target) pairs.

    Each item is a pair of tensors of length `block_size`:
        x = tokens[i   : i+block_size]
        y = tokens[i+1 : i+block_size+1]   (next-token targets)
    """

    def __init__(self, token_ids: list[int], block_size: int):
        self.data       = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — encode an entire HF dataset split into a flat token list
# ─────────────────────────────────────────────────────────────────────────────

def encode_split(hf_split, tokenizer: Tokenizer) -> list[int]:
    """
    Encode every non-empty line in a HuggingFace dataset split.
    Returns a single flat list of token IDs (no padding needed for LM training).
    """
    ids: list[int] = []
    for sample in hf_split:
        text = sample["text"].strip()
        if text:
            ids.extend(tokenizer.encode(text).ids)
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Public factory — returns (train_loader, val_loader, test_loader)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    tokenizer_path: str = "bpe_tokenizer.json",
    block_size:     int = 128,
    batch_size:     int = 32,
    num_workers:    int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Wikitext-2, encode it with the saved BPE tokenizer, and return
    three DataLoaders for train / validation / test.

    Args:
        tokenizer_path: Path to the saved tokenizer JSON (from tokenizer_train.py).
        block_size:     Context window length (must match GPTConfig.block_size).
        batch_size:     Mini-batch size.
        num_workers:    DataLoader worker processes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("Loading tokenizer...")
    tokenizer = TokenizerSingleton.get_instance(tokenizer_path)

    print("Loading Wikitext-2 dataset...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("Encoding splits (this may take ~1 min)...")
    train_ids = encode_split(raw["train"],      tokenizer)
    val_ids   = encode_split(raw["validation"], tokenizer)
    test_ids  = encode_split(raw["test"],       tokenizer)

    print(f"  Train tokens : {len(train_ids):,}")
    print(f"  Val   tokens : {len(val_ids):,}")
    print(f"  Test  tokens : {len(test_ids):,}")

    train_ds = TokenDataset(train_ids, block_size)
    val_ds   = TokenDataset(val_ids,   block_size)
    test_ds  = TokenDataset(test_ids,  block_size)

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader