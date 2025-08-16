import pytest
import json
import os
from tokenizer.vocab import Vocabulary

def test_vocabulary_initialization():
    """Test Vocabulary initialization with custom parameters."""
    vocab = Vocabulary(
        vocab_size=1000,
        unk_token="[UNK]",
        pad_token="[PAD]"
    )
    assert vocab.unk_token == "[UNK]"
    assert vocab.pad_token == "[PAD]"
    assert "[UNK]" in vocab.token2id
    assert "[PAD]" in vocab.token2id

def test_add_token():
    """Test adding tokens to vocabulary."""
    vocab = Vocabulary(vocab_size=1000)
    initial_size = len(vocab.token2id)
    token_id = vocab.add_token("test")
    
    assert "test" in vocab.token2id
    assert vocab.token2id["test"] == token_id
    assert len(vocab.token2id) == initial_size + 1

def test_tokenize():
    """Test tokenization functionality."""
    vocab = Vocabulary(vocab_size=1000)
    # Add some test tokens
    vocab.add_token("hello")
    vocab.add_token("world")
    
    # Test tokenization
    tokens = vocab.tokenize("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0

def test_save_load(tmp_path):
    """Test saving and loading vocabulary."""
    vocab = Vocabulary(vocab_size=1000)
    tokens = ["test", "save", "load"]
    for token in tokens:
        vocab.add_token(token)
    
    # Save to file
    file_path = os.path.join(tmp_path, "vocab.json")
    vocab.save(file_path)
    
    # Verify file exists
    assert os.path.exists(file_path)
    
    # Load from file using class method
    loaded_vocab = Vocabulary.load(file_path)
    
    # Verify basic properties
    assert isinstance(loaded_vocab, Vocabulary)
    assert loaded_vocab.vocab_size == vocab.vocab_size
    
    # Verify tokens
    for token in tokens:
        assert token in loaded_vocab.token2id
        assert loaded_vocab.token2id[token] == vocab.token2id[token]
