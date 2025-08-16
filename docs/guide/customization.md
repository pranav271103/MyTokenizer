# Customization Guide

This guide explains how to customize MyTokenizer to fit your specific needs.

## Table of Contents
- [Custom Tokenization Rules](#custom-tokenization-rules)
- [Custom Vocabulary](#custom-vocabulary)
- [Special Tokens](#special-tokens)
- [Custom Preprocessing](#custom-preprocessing)
- [Custom Postprocessing](#custom-postprocessing)
- [Extending the Tokenizer](#extending-the-tokenizer)

## Custom Tokenization Rules

### Using Regular Expressions

```python
import re
from tokenizer import Tokenizer

# Define custom tokenization pattern
pattern = r"\b\w+\b|\S"  # Words or non-whitespace characters

# Create tokenizer with custom regex
tokenizer = Tokenizer(tokenization_pattern=pattern)
```

### Custom Tokenizer Function

```python
def custom_tokenizer(text):
    # Simple whitespace tokenizer with special handling for contractions
    tokens = []
    for word in text.split():
        if "'" in word:
            # Split contractions like "don't" -> ["do", "n't"]
            parts = word.split("'")
            tokens.extend(parts[:-1])
            tokens.append("'" + parts[-1])
        else:
            tokens.append(word)
    return tokens

# Initialize with custom tokenizer
tokenizer = Tokenizer(custom_tokenizer=custom_tokenizer)
```

## Custom Vocabulary

### Creating a Custom Vocabulary

```python
from tokenizer import Tokenizer, Vocabulary

# Create a vocabulary from a list of tokens
custom_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", 
               "hello", "world", "token", "##ization"]

# Initialize vocabulary
vocab = Vocabulary(custom_vocab)

# Create tokenizer with custom vocabulary
tokenizer = Tokenizer(vocab=vocab)
```

### Updating Vocabulary

```python
# Add new tokens to existing vocabulary
tokenizer.vocab.add_tokens(["new", "tokens", "to", "add"])

# Remove tokens
# Note: Be careful as this might affect existing tokenization
tokenizer.vocab.remove_tokens(["obsolete", "tokens"])
```

## Special Tokens

### Defining Special Tokens

```python
from tokenizer import Tokenizer, TokenizerConfig

config = TokenizerConfig(
    special_tokens={
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]"
    }
)

tokenizer = Tokenizer(config=config)
```

### Adding Special Cases

```python
# Add special cases that should be tokenized in a specific way
tokenizer.add_special_case("gonna", ["gon", "na"])
tokenizer.add_special_case("wanna", ["wan", "na"])
tokenizer.add_special_case("can't", ["can", "n't"])

# Now these will be tokenized as specified
tokens = tokenizer.tokenize("I'm gonna use this tokenizer")
# Output: ["I", "'", "m", "gon", "na", "use", "this", "token", "##izer"]
```

## Custom Preprocessing

### Preprocessing Pipeline

```python
def custom_preprocessor(text):
    """Custom preprocessing function."""
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'(@\w+|#\w+)', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Initialize with custom preprocessor
tokenizer = Tokenizer(preprocessor=custom_preprocessor)
```

### Multiple Preprocessors

```python
from functools import reduce

def compose(*functions):
    """Compose multiple functions into a single function."""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

def remove_punctuation(text):
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

# Combine multiple preprocessing functions
preprocessing_pipeline = compose(
    custom_preprocessor,
    remove_punctuation,
    remove_numbers
)

tokenizer = Tokenizer(preprocessor=preprocessing_pipeline)
```

## Custom Postprocessing

### Postprocessing Tokens

```python
def custom_postprocessor(tokens):
    """Custom postprocessing of tokens."""
    # Remove empty strings
    tokens = [token for token in tokens if token.strip()]
    
    # Merge certain tokens
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == "##" and i + 1 < len(tokens):
            tokens[i:i+2] = [tokens[i] + tokens[i+1]]
        else:
            i += 1
    
    return tokens

# Initialize with custom postprocessor
tokenizer = Tokenizer(postprocessor=custom_postprocessor)
```

## Extending the Tokenizer

### Creating a Custom Tokenizer Class

```python
from tokenizer import Tokenizer

class CustomTokenizer(Tokenizer):
    """Custom tokenizer with additional functionality."""
    
    def __init__(self, *args, **kwargs):
        # Custom initialization
        self.custom_param = kwargs.pop('custom_param', None)
        super().__init__(*args, **kwargs)
    
    def tokenize(self, text, **kwargs):
        # Custom tokenization logic
        if self.custom_param:
            # Apply custom preprocessing
            text = self._preprocess_with_custom_param(text)
        
        # Call parent's tokenize method
        tokens = super().tokenize(text, **kwargs)
        
        # Apply custom postprocessing
        return self._custom_postprocess(tokens)
    
    def _preprocess_with_custom_param(self, text):
        # Custom preprocessing logic
        return text
    
    def _custom_postprocess(self, tokens):
        # Custom postprocessing logic
        return tokens

# Use the custom tokenizer
custom_tokenizer = CustomTokenizer(custom_param="value")
```

## Next Steps

- [Advanced Usage](./advanced-usage.md) - Learn about advanced features
- [API Reference](../api/tokenizer.md) - Detailed API documentation
- [Training Examples](../examples/training.md) - Examples for training tokenizers
- [Fine-tuning Examples](../examples/finetuning.md) - Examples for fine-tuning models
