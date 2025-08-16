# Basic Usage

This guide covers the fundamental usage patterns of MyTokenizer.

## Table of Contents
- [Tokenization](#tokenization)
- [Special Tokens](#special-tokens)
- [Configuration](#configuration)
- [Handling Different Languages](#handling-different-languages)
- [Error Handling](#error-handling)

## Tokenization

### Basic Tokenization

```python
from tokenizer import Tokenizer

# Initialize with default settings
tokenizer = Tokenizer()

# Tokenize a simple text
text = "MyTokenizer makes tokenization easy and efficient."
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['My', '##Token', '##izer', 'makes', 'token', '##ization', 'easy', 'and', 'efficient', '.']
```

### Batch Processing

Process multiple texts efficiently:

```python
texts = [
    "First example text.",
    "Second example with different length.",
    "Third example for batch processing."
]

batch_tokens = tokenizer.tokenize_batch(texts)
for i, tokens in enumerate(batch_tokens):
    print(f"Text {i+1}:", tokens)
```

## Special Tokens

### Using Special Tokens

```python
from tokenizer import Tokenizer, TokenizerConfig

config = TokenizerConfig(
    special_tokens={
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    }
)

tokenizer = Tokenizer(config=config)
```

### Adding Special Tokens After Initialization

```python
tokenizer.add_special_tokens(["[NEW1]", "[NEW2]"])
```

## Configuration

### Custom Configuration

```python
from tokenizer import Tokenizer, TokenizerConfig

config = TokenizerConfig(
    algorithm="bpe",          # or "wordpiece", "unigram"
    vocab_size=30000,
    min_frequency=2,
    lowercase=True,
    strip_accents=True,
    max_length=512,
    truncation=True,
    padding="max_length"
)

tokenizer = Tokenizer(config=config)
```

### Available Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| algorithm | str | "bpe" | Tokenization algorithm (bpe/wordpiece/unigram) |
| vocab_size | int | 30000 | Maximum size of the vocabulary |
| min_frequency | int | 2 | Minimum frequency for a token to be included |
| lowercase | bool | True | Whether to convert text to lowercase |
| strip_accents | bool | True | Whether to strip accents |
| max_length | int | 512 | Maximum sequence length |
| truncation | bool | True | Whether to truncate sequences |
| padding | str | "max_length" | Padding strategy |
| special_tokens | dict | {} | Special tokens configuration |

## Handling Different Languages

### Multilingual Tokenization

```python
# Initialize with language-specific settings
config = TokenizerConfig(
    language="en",  # Supports multiple languages
    lowercase=True,
    strip_accents=True
)

tokenizer = Tokenizer(config=config)

# Tokenize text in different languages
text_en = "This is an English text."
text_es = "Este es un texto en español."

tokens_en = tokenizer.tokenize(text_en)
tokens_es = tokenizer.tokenize(text_es)
```

## Error Handling

### Basic Error Handling

```python
try:
    tokens = tokenizer.tokenize(None)
except ValueError as e:
    print(f"Error: {e}")
```

### Handling Unknown Tokens

```python
text = "This word_is_unknown_to_the_tokenizer"
tokens = tokenizer.tokenize(text, handle_unknown=True)
# Unknown words will be replaced with the unknown token
```

## Next Steps

- [Advanced Usage](./advanced-usage.md) - Learn about advanced features
- [Customization](./customization.md) - Customize tokenizer behavior
- [API Reference](../api/tokenizer.md) - Detailed API documentation
