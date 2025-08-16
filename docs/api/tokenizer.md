# Tokenizer API Reference

This document provides detailed documentation for the `Tokenizer` class, which is the main class for tokenizing text using various algorithms.

## Class: Tokenizer

```python
class tokenizer.Tokenizer(config=None, **kwargs)
```

Main class for tokenizing text. This class can be configured to use different tokenization algorithms (BPE, WordPiece, Unigram) and supports various customization options.

### Parameters

- **config** (`TokenizerConfig`, *optional*) - Configuration object for the tokenizer. If not provided, default settings will be used.
- **\*\*kwargs** - Additional keyword arguments that will be used to update the configuration.

### Attributes

- **vocab** (`Vocabulary`) - The vocabulary used by the tokenizer.
- **config** (`TokenizerConfig`) - The configuration object for the tokenizer.
- **preprocessor** (`callable`, *optional*) - Function to preprocess text before tokenization.
- **postprocessor** (`callable`, *optional*) - Function to postprocess tokens after tokenization.

### Methods

#### `tokenize`

```python
tokenize(text, **kwargs)
```

Tokenize the input text.

**Parameters:**
- **text** (`str`) - The text to tokenize.
- **\*\*kwargs** - Additional arguments to override tokenizer configuration.

**Returns:**
- `List[str]` - List of tokens.

**Example:**
```python
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("Hello, world!")
# Returns: ["Hello", ",", "world", "!"]
```

#### `tokenize_batch`

```python
tokenize_batch(texts, **kwargs)
```

Tokenize a batch of texts.

**Parameters:**
- **texts** (`List[str]`) - List of texts to tokenize.
- **\*\*kwargs** - Additional arguments to override tokenizer configuration.

**Returns:**
- `List[List[str]]` - List of token lists, one for each input text.

**Example:**
```python
tokenizer = Tokenizer()
texts = ["First text", "Second text"]
batch_tokens = tokenizer.tokenize_batch(texts)
# Returns: [["First", "text"], ["Second", "text"]]
```

#### `train`

```python
train(files, vocab_size=30000, min_frequency=2, show_progress=True, **kwargs)
```

Train the tokenizer on the given files.

**Parameters:**
- **files** (`str` or `List[str]`) - File path or list of file paths to train on.
- **vocab_size** (`int`, *optional*, defaults to 30000) - Maximum size of the vocabulary.
- **min_frequency** (`int`, *optional*, defaults to 2) - Minimum frequency for a token to be included.
- **show_progress** (`bool`, *optional*, defaults to `True`) - Whether to show a progress bar.
- **\*\*kwargs** - Additional training parameters.

**Example:**
```python
tokenizer = Tokenizer()
tokenizer.train("data.txt", vocab_size=50000)
```

#### `save`

```python
save(file_path)
```

Save the tokenizer to a file.

**Parameters:**
- **file_path** (`str`) - Path where to save the tokenizer.

**Example:**
```python
tokenizer.save("models/my_tokenizer.model")
```

#### `load`

```python
@classmethod
load(file_path)
```

Load a tokenizer from a file.

**Parameters:**
- **file_path** (`str`) - Path to the saved tokenizer file.

**Returns:**
- `Tokenizer` - The loaded tokenizer instance.

**Example:**
```python
tokenizer = Tokenizer.load("models/pretrained.model")
```

#### `add_special_tokens`

```python
add_special_tokens(tokens)
```

Add special tokens to the vocabulary.

**Parameters:**
- **tokens** (`List[str]` or `str`) - Token or list of tokens to add.

**Example:**
```python
tokenizer.add_special_tokens(["[NEW1]", "[NEW2]"])
```

#### `add_special_case`

```python
add_special_case(token, tokens_sequence)
```

Add a special case for tokenization.

**Parameters:**
- **token** (`str`) - The token to replace.
- **tokens_sequence** (`List[str]`) - The sequence of tokens to replace it with.

**Example:**
```python
tokenizer.add_special_case("gonna", ["gon", "na"])
```

#### `enable_cache`

```python
enable_cache(max_size=10000)
```

Enable caching of tokenization results.

**Parameters:**
- **max_size** (`int`, *optional*, defaults to 10000) - Maximum number of items to cache.

#### `disable_cache`

```python
disable_cache()
```

Disable caching of tokenization results.

#### `clear_cache`

```python
clear_cache()
```

Clear the tokenization cache.

### Properties

#### `vocab_size`

```python
@property
vocab_size()
```

Get the size of the vocabulary.

**Returns:**
- `int` - Number of tokens in the vocabulary.

#### `special_tokens`

```python
@property
special_tokens()
```

Get the list of special tokens.

**Returns:**
- `List[str]` - List of special tokens.

### Class Methods

#### `from_pretrained`

```python
@classmethod
from_pretrained(model_name_or_path, **kwargs)
```

Load a pretrained tokenizer.

**Parameters:**
- **model_name_or_path** (`str`) - Name or path of the pretrained model.
- **\*\*kwargs** - Additional arguments for loading the tokenizer.

**Returns:**
- `Tokenizer` - The loaded tokenizer instance.

**Example:**
```python
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
```

#### `get_config`

```python
@classmethod
get_config()
```

Get the default configuration for the tokenizer.

**Returns:**
- `dict` - Default configuration dictionary.

### Configuration

The `Tokenizer` class can be configured using a `TokenizerConfig` object. Here are the available configuration options:

```python
config = TokenizerConfig(
    algorithm="bpe",          # or "wordpiece", "unigram"
    vocab_size=30000,         # Maximum vocabulary size
    min_frequency=2,          # Minimum token frequency
    lowercase=True,           # Convert to lowercase
    strip_accents=True,       # Strip accents
    max_length=512,           # Maximum sequence length
    truncation=True,          # Whether to truncate sequences
    padding="max_length",     # Padding strategy
    special_tokens={
        "unk_token": "[UNK]",
        "pad_token": "[PAD]"
    }
)
```

### Examples

#### Basic Usage

```python
from tokenizer import Tokenizer

# Initialize with default settings
tokenizer = Tokenizer()

# Tokenize text
tokens = tokenizer.tokenize("Hello, world!")
print(tokens)
# Output: ["Hello", ",", "world", "!"]
```

#### Custom Configuration

```python
from tokenizer import Tokenizer, TokenizerConfig

# Custom configuration
config = TokenizerConfig(
    algorithm="wordpiece",
    vocab_size=50000,
    lowercase=True,
    special_tokens={
        "unk_token": "[UNK]",
        "pad_token": "[PAD]"
    }
)

# Initialize with custom configuration
tokenizer = Tokenizer(config=config)
```

#### Training a Tokenizer

```python
from tokenizer import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer()

# Train on a text file
tokenizer.train("data.txt", vocab_size=50000)

# Save the trained tokenizer
tokenizer.save("my_tokenizer.model")

# Load the tokenizer later
loaded_tokenizer = Tokenizer.load("my_tokenizer.model")
```

### Notes

- The tokenizer automatically handles unknown tokens by replacing them with the `[UNK]` token.
- Special tokens are never split during tokenization.
- The tokenizer can be extended with custom preprocessors and postprocessors.
