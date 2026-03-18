"""
EECE 4520 - Milestone 3 Part 1
Step 1: Train a BPE tokenizer on Wikitext-2 and save it.
Run this FIRST before anything else.
"""
 
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
 
 
def get_corpus_iterator(dataset):
    """Yield raw text strings from the dataset for tokenizer training."""
    for split in ["train", "validation", "test"]:
        for sample in dataset[split]:
            text = sample["text"].strip()
            if text:
                yield text
 
 
def train_bpe_tokenizer(vocab_size: int = 8000, save_path: str = "bpe_tokenizer.json"):
    """
    Train a Byte-Pair Encoding tokenizer on Wikitext-2.
 
    Args:
        vocab_size: Number of tokens in the vocabulary.
        save_path:  Where to save the trained tokenizer JSON.
    """
    print("Loading Wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
 
    # --- Build tokenizer with BPE model ---
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
 
    # Special tokens
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
 
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )
 
    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer.train_from_iterator(
        get_corpus_iterator(dataset),
        trainer=trainer,
    )
 
    # Add post-processor so [BOS]/[EOS] are inserted automatically
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
 
    tokenizer.save(save_path)
    print(f"Tokenizer saved to '{save_path}'")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
 
    # Quick sanity check
    sample = "The transformer architecture revolutionized natural language processing."
    encoded = tokenizer.encode(sample)
    print(f"\nSample encode → decode check:")
    print(f"  Input  : {sample}")
    print(f"  Tokens : {encoded.tokens[:12]} ...")
    print(f"  IDs    : {encoded.ids[:12]} ...")
    print(f"  Decoded: {tokenizer.decode(encoded.ids)}")
 
 
if __name__ == "__main__":
    train_bpe_tokenizer(vocab_size=8000, save_path="bpe_tokenizer.json")
 