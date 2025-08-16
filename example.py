"""
Example script demonstrating how to train and use the custom tokenizer.
"""

import os
import logging
from tokenizer import Tokenizer, TokenizerTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_tokenizer():
    """Train a tokenizer on sample text."""
    # Sample text data (in practice, you would load this from files)
    sample_texts = [
        "This is an example sentence for tokenizer training.",
        "The tokenizer will learn to split text into subword units.",
        "It can handle various languages and special characters.",
        "The training process builds a vocabulary of the most common subwords.",
        "This helps in handling out-of-vocabulary words effectively.",
    ]

    # Save sample data to a temporary file
    temp_file = "sample_data.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_texts))

    try:
        # Initialize trainer
        trainer = TokenizerTrainer(vocab_size=1000, min_frequency=1, lowercase=True)

        # Train tokenizer
        logger.info("Starting tokenizer training...")
        tokenizer = trainer.train(
            files=[temp_file],
            algorithm="bpe",
            num_workers=1,  # Use 1 worker for small dataset
        )

        # Save tokenizer
        os.makedirs("trained_tokenizer", exist_ok=True)
        trainer.save("trained_tokenizer")
        logger.info("Tokenizer training completed and saved to 'trained_tokenizer'")

        return tokenizer

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def load_and_use_tokenizer():
    """Load a trained tokenizer and use it for tokenization."""
    # First ensure we have a trained tokenizer
    tokenizer = train_tokenizer()

    # Load trained tokenizer
    trainer = TokenizerTrainer.load("trained_tokenizer")
    tokenizer = trainer.tokenizer
    logger.info("Loaded pre-trained tokenizer")

    # Example usage
    test_text = "This is a test sentence with some unknown words like 'tokenization' and 'subwords'."

    # Tokenize text
    tokens = tokenizer.tokenize(test_text)
    token_ids = [tokenizer.vocab[token] for token in tokens]

    print("\nTokenization Example:")
    print(f"Original text: {test_text}")
    print(f"\nTokens:")
    for token in tokens:
        print(f"  {token}")

    print(f"\nToken IDs:")
    for token, token_id in zip(tokens, token_ids):
        print(f"  {token}: {token_id}")

    # Show vocabulary size
    print(f"\nVocabulary size: {len(tokenizer.vocab)}")

    # Show more vocabulary items
    print("\nSample vocabulary items:")
    for i, (token, token_id) in enumerate(list(tokenizer.vocab.token2id.items())[:20]):
        print(f"{token}: {token_id}")

    # Show tokenization of specific words
    print("\nTokenizing specific words:")
    test_words = ["tokenization", "subwords", "unknown"]
    for word in test_words:
        print(f"\nWord: {word}")
        tokens = tokenizer.tokenize(word)
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {[tokenizer.vocab[token] for token in tokens]}")


if __name__ == "__main__":
    load_and_use_tokenizer()
