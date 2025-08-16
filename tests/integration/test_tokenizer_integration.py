import pytest
from tokenizer.tokenizer import Tokenizer
from tokenizer.trainer import TokenizerTrainer
import tempfile
import os

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
            min_frequency=1,
            show_progress=False
        )
        tokenizer = trainer.train_from_files([sample_text_file])
        
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
            min_frequency=1,
            show_progress=False
        )
        tokenizer = trainer.train_from_files([sample_text_file])
        
        # Save tokenizer
        save_path = tmp_path / "test_tokenizer.json"
        tokenizer.save(save_path)
        
        # Load tokenizer
        loaded_tokenizer = Tokenizer.from_file(save_path)
        
        # Test loaded tokenizer
        text = "This is a test sentence."
        original_tokens = tokenizer.tokenize(text)
        loaded_tokens = loaded_tokenizer.tokenize(text)
        
        assert original_tokens == loaded_tokens

    def test_special_tokens_handling(self):
        """Test that special tokens are handled correctly."""
        tokenizer = Tokenizer(
            special_tokens={
                "unk_token": "[UNK]",
                "pad_token": "[PAD]"
            }
        )
        
        # Test that special tokens are in the vocabulary
        assert "[UNK]" in tokenizer.vocab
        assert "[PAD]" in tokenizer.vocab
        
        # Test that unknown tokens are replaced with UNK
        tokens = tokenizer.tokenize("This is an unknownword")
        assert "[UNK]" in tokens
