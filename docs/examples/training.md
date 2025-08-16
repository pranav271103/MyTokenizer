# Training Examples

This guide provides examples of training custom tokenizers using different algorithms and configurations.

## Table of Contents
- [Training a BPE Tokenizer](#training-a-bpe-tokenizer)
- [Training a WordPiece Tokenizer](#training-a-wordpiece-tokenizer)
- [Training a Unigram Tokenizer](#training-a-unigram-tokenizer)
- [Training on Multiple Files](#training-on-multiple-files)
- [Incremental Training](#incremental-training)
- [Custom Training Callbacks](#custom-training-callbacks)

## Training a BPE Tokenizer

```python
from tokenizer import Tokenizer, TokenizerConfig

# Define configuration
config = TokenizerConfig(
    algorithm="bpe",
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    lowercase=True,
    max_length=512
)

# Initialize tokenizer
tokenizer = Tokenizer(config=config)

# Train on a text file
tokenizer.train("data/train.txt", vocab_size=30000)

# Save the trained tokenizer
tokenizer.save("models/bpe_tokenizer.model")
```

## Training a WordPiece Tokenizer

```python
from tokenizer import Tokenizer, TokenizerConfig

# Define configuration
config = TokenizerConfig(
    algorithm="wordpiece",
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    lowercase=True,
    wordpieces_prefix="##"
)

# Initialize tokenizer
tokenizer = Tokenizer(config=config)

# Train on a text file
tokenizer.train("data/train.txt", vocab_size=30000)

# Save the trained tokenizer
tokenizer.save("models/wordpiece_tokenizer.model")
```

## Training a Unigram Tokenizer

```python
from tokenizer import Tokenizer, TokenizerConfig

# Define configuration
config = TokenizerConfig(
    algorithm="unigram",
    vocab_size=30000,
    unk_token="[UNK]",
    special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    shrinking_factor=0.75
)

# Initialize tokenizer
tokenizer = Tokenizer(config=config)

# Train on a text file
tokenizer.train("data/train.txt", vocab_size=30000)

# Save the trained tokenizer
tokenizer.save("models/unigram_tokenizer.model")
```

## Training on Multiple Files

```python
from tokenizer import Tokenizer, TokenizerConfig

# List of training files
training_files = [
    "data/train_part1.txt",
    "data/train_part2.txt",
    "data/additional_data.txt"
]

# Initialize tokenizer
tokenizer = Tokenizer()

# Train on multiple files
tokenizer.train(
    files=training_files,
    vocab_size=50000,
    min_frequency=2,
    show_progress=True
)

# Save the trained tokenizer
tokenizer.save("models/multi_file_tokenizer.model")
```

## Incremental Training

```python
from tokenizer import Tokenizer

# Load existing tokenizer
tokenizer = Tokenizer.load("models/pretrained_tokenizer.model")

# Continue training with new data
tokenizer.train(
    "data/new_data.txt",
    vocab_size=55000,  # Optionally increase vocabulary size
    min_frequency=2,
    show_progress=True
)

# Save the updated tokenizer
tokenizer.save("models/updated_tokenizer.model")
```

## Custom Training Callbacks

```python
from tokenizer import Tokenizer, TokenizerConfig

# Define callbacks
class TrainingCallbacks:
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}. Vocab size: {logs.get('vocab_size')}")
    
    def on_batch_end(self, batch, logs=None):
        if batch % 1000 == 0:
            print(f"Processed {batch} batches")

# Initialize tokenizer with callbacks
tokenizer = Tokenizer()
callbacks = TrainingCallbacks()

# Train with callbacks
tokenizer.train(
    "data/large_corpus.txt",
    vocab_size=50000,
    callbacks=callbacks,
    batch_size=1000
)

# Save the trained tokenizer
tokenizer.save("models/callback_tokenizer.model")
```

## Training with Custom Preprocessing

```python
from tokenizer import Tokenizer, TokenizerConfig
import re

def custom_preprocessor(text):
    # Custom preprocessing function
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Initialize tokenizer with custom preprocessor
config = TokenizerConfig(
    algorithm="bpe",
    vocab_size=30000,
    preprocessor=custom_preprocessor
)

tokenizer = Tokenizer(config=config)

# Train with custom preprocessing
tokenizer.train("data/raw_text.txt")

# Save the trained tokenizer
tokenizer.save("models/preprocessed_tokenizer.model")
```

## Training with Limited Resources

```python
from tokenizer import Tokenizer, TokenizerConfig

# Configure for limited memory usage
config = TokenizerConfig(
    algorithm="bpe",
    vocab_size=20000,  # Smaller vocabulary
    memory_limit="2GB",  # Limit memory usage
    batch_size=1000,     # Smaller batch size
    lowercase=True
)

tokenizer = Tokenizer(config=config)

# Train with limited resources
tokenizer.train(
    "data/large_corpus.txt",
    show_progress=True
)

# Save the trained tokenizer
tokenizer.save("models/lightweight_tokenizer.model")
```

## Next Steps

- [Basic Usage](../guide/basic-usage.md) - Learn the basics of using the tokenizer
- [Advanced Usage](../guide/advanced-usage.md) - Explore advanced features
- [API Reference](../api/tokenizer.md) - Detailed API documentation
