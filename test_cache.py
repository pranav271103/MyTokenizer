import logging
from tokenizer.vocab import Vocabulary
from tokenizer.tokenizer import Tokenizer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_cache():
    """Test the cache functionality directly."""
    # Create a simple vocabulary
    vocab = Vocabulary()

    # Add some tokens
    tokens = ["hello", "world", "test", "example"]
    for token in tokens:
        vocab.add_token(token)

    # Train vocabulary using BPE algorithm
    logger.info("Training vocabulary using BPE...")
    vocab.train_on_texts(tokens, algorithm="bpe", num_merges=100)
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Create tokenizer
    tokenizer = Tokenizer(vocab)

    # Test text
    test_text = "hello world test example"

    # First tokenization (should be cache miss)
    logger.info("\nFirst tokenization (should be cache miss)")
    logger.info(f"Input text: '{test_text}'")
    result1 = tokenizer.tokenize(test_text)
    logger.info(f"Result: {result1}")

    # Second tokenization (should be cache hit)
    logger.info("\nSecond tokenization (should be cache hit)")
    logger.info(f"Input text: '{test_text}'")
    result2 = tokenizer.tokenize(test_text)
    logger.info(f"Result: {result2}")

    # Get cache stats
    cache_hits, cache_misses = vocab.get_cache_stats()
    logger.info(f"\nCache Stats:")
    logger.info(f"Hits: {cache_hits}")
    logger.info(f"Misses: {cache_misses}")
    logger.info(
        f"Hit Rate: {cache_hits / (cache_hits + cache_misses) if cache_hits + cache_misses > 0 else 0:.2%}"
    )

    # Check vocabulary contents
    logger.info("\nVocabulary contents:")
    for token in tokens:
        token_id = vocab[token]  # Use __getitem__ to get token ID
        logger.info(f"Token '{token}' ID: {token_id}")


if __name__ == "__main__":
    test_cache()
