# Installation

## Using pip

```bash
pip install ultratokenizer
```

## From Source

```bash
git clone https://github.com/pranav271103/Ultra-Tokenizer.git
cd Ultra-Tokenizer
pip install -e .
```

# Quickstart

```python
from tokenizer import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer()

# Tokenize text
tokens = tokenizer.tokenize("Hello, world!")
print(tokens)
```