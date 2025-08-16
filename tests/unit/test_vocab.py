import pytest
from tokenizer.vocab import Vocabulary

def test_vocabulary_initialization():
    """Test Vocabulary initialization with custom parameters."""
    vocab = Vocabulary(
        unk_token="[UNK]",
        pad_token="[PAD]"
    )
    assert vocab.unk_token == "[UNK]"
    assert vocab.pad_token == "[PAD]"
    assert vocab.unk_token_id == 0
    assert vocab.pad_token_id == 1

def test_add_token():
    """Test adding tokens to vocabulary."""
    vocab = Vocabulary()
    token_id = vocab.add_token("test")
    assert token_id == 2  # After [UNK] and [PAD]
    assert "test" in vocab
    assert vocab["test"] == token_id

def test_encode_decode():
    """Test encoding and decoding tokens."""
    vocab = Vocabulary()
    tokens = ["hello", "world", "!"]
    
    # Add tokens
    for token in tokens:
        vocab.add_token(token)
    
    # Test encoding
    encoded = vocab.encode(tokens)
    assert len(encoded) == len(tokens)
    
    # Test decoding
    decoded = vocab.decode(encoded)
    assert decoded == tokens

def test_save_load(tmp_path):
    """Test saving and loading vocabulary."""
    vocab = Vocabulary()
    tokens = ["test", "save", "load"]
    for token in tokens:
        vocab.add_token(token)
    
    # Save to file
    file_path = tmp_path / "vocab.json"
    vocab.save(file_path)
    
    # Load from file
    loaded_vocab = Vocabulary.from_file(file_path)
    
    # Verify
    assert len(loaded_vocab) == len(vocab)
    for token in tokens:
        assert token in loaded_vocab
        assert loaded_vocab[token] == vocab[token]
