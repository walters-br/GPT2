from tokenizers import Tokenizer


class TokenizerSingleton:
  """
  Singleton wrapper around a HuggingFace BPE Tokenizer.

  Ensures that the tokenizer is loaded from disk only once and shared across the entire application.
  """

  _instance: "TokenizerSingleton | None" = None  # class-level shared reference

  def __init__(self, tokenizer_path: str):
    """Private"""
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    self._tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)

  # Singleton access point

  @classmethod
  def get_instance(cls, tokenizer_path: str = "bpe_tokenizer.json") -> Tokenizer:
    """Return the shared tokenizer, loading it from disk if this is the first call."""
    if cls._instance is None:
      cls._instance = cls(tokenizer_path)
    return cls._instance._tokenizer

  @classmethod
  def reset(cls) -> None:
    """Clear the singleton."""
    cls._instance = None