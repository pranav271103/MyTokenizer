# Advanced Usage

This guide covers advanced features and techniques for using MyTokenizer effectively.

## Table of Contents
- [Custom Tokenization](#custom-tokenization)
- [Training Custom Models](#training-custom-models)
- [Performance Optimization](#performance-optimization)
- [Custom Preprocessing](#custom-preprocessing)
- [Parallel Processing](#parallel-processing)
- [Memory Management](#memory-management)

## Custom Tokenization

### Custom Tokenization Rules

```python
from tokenizer import Tokenizer, TokenizerConfig
import re

def custom_tokenizer(text):
    # Split on whitespace and punctuation, but keep email addresses intact
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|\w+|\S"
    return re.findall(pattern, text)

config = TokenizerConfig(
    custom_tokenizer=custom_tokenizer,
    lowercase=True
)

tokenizer = Tokenizer(config=config)
```

### Handling Special Cases

```python
# Add special cases to the tokenizer
tokenizer.add_special_case("gonna", ["gon", "na"])
tokenizer.add_special_case("wanna", ["wan", "na"])

# Now these will be tokenized as specified
tokens = tokenizer.tokenize("I'm gonna use this tokenizer")
# Output: ["I", "'", "m", "gon", "na", "use", "this", "token", "##izer"]
```

## Training Custom Models

### Training on Custom Data

```python
from tokenizer import Tokenizer, TokenizerConfig

# Prepare your training data
corpus_files = ["data/train.txt", "data/additional.txt"]

# Configure training
config = TokenizerConfig(
    algorithm="bpe",
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Initialize and train
tokenizer = Tokenizer(config=config)
tokenizer.train(
    files=corpus_files,
    vocab_size=50000,
    min_frequency=2,
    show_progress=True
)

# Save the trained tokenizer
tokenizer.save("models/custom_tokenizer.model")
```

### Incremental Training

```python
# Load existing tokenizer
tokenizer = Tokenizer.load("models/pretrained.model")

# Continue training with new data
tokenizer.train(
    files=["data/new_data.txt"],
    vocab_size=55000,  # Optionally increase vocabulary size
    min_frequency=2
)
```

## Performance Optimization

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", ...]  # Large list of texts
batch_size = 1000

# Process in batches
all_tokens = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    tokens = tokenizer.tokenize_batch(batch)
    all_tokens.extend(tokens)
```

### Caching

```python
# Enable caching for repeated texts
tokenizer.enable_cache(max_size=10000)  # Cache up to 10,000 unique texts

# First call is slower (computes tokens)
tokens1 = tokenizer.tokenize("This text will be cached.")

# Subsequent calls are faster (uses cache)
tokens2 = tokenizer.tokenize("This text will be cached.")

# Clear cache if needed
tokenizer.clear_cache()
```

## Custom Preprocessing

### Custom Preprocessing Pipeline

```python
from tokenizer import Tokenizer
import re

def custom_preprocessor(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'(@\w+|#\w+)', '', text)
    return text.strip()

# Initialize with custom preprocessor
tokenizer = Tokenizer(preprocessor=custom_preprocessor)
```

### Using Multiple Preprocessors

```python
def remove_emojis(text):
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    # Apply multiple cleaning steps
    text = custom_preprocessor(text)
    text = remove_emojis(text)
    return text

tokenizer = Tokenizer(preprocessor=clean_text)
```

## Parallel Processing

### Using Multiple Cores

```python
from tokenizer import Tokenizer
from multiprocessing import cpu_count

# Initialize tokenizer with parallel processing
tokenizer = Tokenizer(n_jobs=cpu_count())  # Use all available cores

# Process large datasets in parallel
texts = [...]  # Large list of texts
tokens = tokenizer.tokenize_batch(texts)  # Will use multiple cores
```

## Memory Management

### Controlling Memory Usage

```python
# Limit memory usage during training
config = TokenizerConfig(
    algorithm="bpe",
    vocab_size=50000,
    memory_limit="4GB"  # Limit memory usage to 4GB
)

tokenizer = Tokenizer(config=config)
tokenizer.train("large_corpus.txt")
```

### Streaming Large Files

```python
def text_generator(file_path, batch_size=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Process file in batches without loading everything into memory
for batch in text_generator("very_large_file.txt"):
    tokens = tokenizer.tokenize_batch(batch)
    # Process tokens...
```

## Next Steps

- [Customization](./customization.md) - Learn how to customize tokenizer behavior
- [API Reference](../api/tokenizer.md) - Detailed API documentation
- [Training Examples](../examples/training.md) - Examples for training tokenizers
- [Fine-tuning Examples](../examples/finetuning.md) - Examples for fine-tuning models
