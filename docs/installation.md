# Installation

## Using pip

```bash
pip install mytokenizer
```

## From Source

```bash
git clone https://github.com/pranav271103/MyTokenizer.git
cd MyTokenizer
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