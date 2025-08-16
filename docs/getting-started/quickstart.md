# Quickstart Guide

This guide will help you get started with MyTokenizer quickly. We'll cover the basic usage patterns and common operations.

## Basic Usage

### Importing the Tokenizer

```python
from tokenizer import Tokenizer
```

### Initializing the Tokenizer

Create a tokenizer instance with default settings:

```python
tokenizer = Tokenizer()
```

### Tokenizing Text

Tokenize a simple sentence:

```python
text = "Hello, world! This is MyTokenizer in action."
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'My', '##Token', '##izer', 'in', 'action', '.']
```

### Batch Processing

Process multiple texts at once:

```python
texts = [
    "First sentence to tokenize.",
    "Second sentence for demonstration.",
    "Third example shows batch processing."
]

all_tokens = [tokenizer.tokenize(text) for text in texts]
print(all_tokens)
```

## Working with Different Algorithms

### Using BPE (Byte Pair Encoding)

```python
from tokenizer import Tokenizer, TokenizerConfig

config = TokenizerConfig(
    algorithm="bpe",
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

bpe_tokenizer = Tokenizer(config=config)
```

### Using WordPiece

```python
config = TokenizerConfig(
    algorithm="wordpiece",
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

wordpiece_tokenizer = Tokenizer(config=config)
```

## Training Your Own Tokenizer

### Prepare Training Data

Create a text file with your training data (one sentence per line):

```python
with open("training_data.txt", "w", encoding="utf-8") as f:
    f.write("This is the first training sentence.\n")
    f.write("Here's another example for the tokenizer.\n")
    # Add more training data...
```

### Train the Tokenizer

```python
from tokenizer import Tokenizer, TokenizerConfig

config = TokenizerConfig(
    algorithm="bpe",
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer = Tokenizer(config=config)
tokenizer.train("training_data.txt")
```

## Saving and Loading

### Save the Tokenizer

```python
tokenizer.save("my_tokenizer.model")
```

### Load the Tokenizer

```python
from tokenizer import Tokenizer

loaded_tokenizer = Tokenizer.load("my_tokenizer.model")
```

## Next Steps

- Explore [Advanced Usage](../guide/advanced-usage.md) for more features
- Check out the [API Reference](../api/tokenizer.md) for detailed documentation
