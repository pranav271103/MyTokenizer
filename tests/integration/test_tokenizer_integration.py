import pytest
import os
import tempfile
from tokenizer.tokenizer import Tokenizer
from tokenizer.trainer import TokenizerTrainer

class TestTokenizerIntegration:
    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary file with sample text for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is a test file.\nIt contains multiple lines.\nFor testing purposes.")
        yield f.name
        os.unlink(f.name)

    def test_train_and_tokenize(self, sample_text_file):
        """Test training a tokenizer and then using it to tokenize text."""
        # Initialize and train tokenizer
        trainer = TokenizerTrainer(
            vocab_size=1000,
            min_frequency=1
        )
        
        # Train on the sample file
        tokenizer = trainer.train([sample_text_file])
        
        # Test tokenization
        text = "This is a test sentence."
        tokens = tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert all(isinstance(token, str) for token in tokens)
        assert len(tokens) > 0

    def test_save_and_load(self, sample_text_file, tmp_path):
        """Test saving and loading a trained tokenizer."""
        # Train a tokenizer
        trainer = TokenizerTrainer(
            vocab_size=1000,
            min_frequency=1
        )
        tokenizer = trainer.train([sample_text_file])
        
        # Create a directory for saving the tokenizer
        save_dir = os.path.join(tmp_path, "test_tokenizer")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save tokenizer - this will create files inside save_dir
        trainer.save(save_dir)
        
        # Load tokenizer using TokenizerTrainer.load()
        loaded_trainer = TokenizerTrainer.load(save_dir)
        loaded_tokenizer = loaded_trainer.tokenizer
        
        # Test loaded tokenizer
        text = "This is a test sentence."
        original_tokens = tokenizer.tokenize(text)
        loaded_tokens = loaded_tokenizer.tokenize(text)
        
        assert original_tokens == loaded_tokens, \
            f"Tokens don't match. Original: {original_tokens}, Loaded: {loaded_tokens}"
        
        # Also test loading using Tokenizer.load() for backward compatibility
        tokenizer_save_dir = os.path.join(tmp_path, "tokenizer_only")
        os.makedirs(tokenizer_save_dir, exist_ok=True)
        tokenizer.save(tokenizer_save_dir)
        
        loaded_tokenizer2 = Tokenizer.load(tokenizer_save_dir)
        loaded_tokens2 = loaded_tokenizer2.tokenize(text)
        assert original_tokens == loaded_tokens2, \
            f"Tokens don't match. Original: {original_tokens}, Loaded2: {loaded_tokens2}"

    def test_special_tokens_handling(self):
        """Test that special tokens are handled correctly."""
        # Initialize tokenizer with default special tokens
        tokenizer = Tokenizer()
        
        # Test that default special tokens are in the vocabulary
        assert "[UNK]" in tokenizer.vocab.token2id
        assert "[PAD]" in tokenizer.vocab.token2id
        
        # Test that unknown tokens are handled
        tokens = tokenizer.tokenize("This is an unknownword")
        assert len(tokens) > 0  # Should not raise an exception
