# MyTokenizer

<div class="grid cards" markdown>

-   :fontawesome-brands-python: **Python 3.8+**
    ---
    Works with all modern Python versions

-   :material-speedometer: **High Performance**
    ---
    Optimized for speed and memory efficiency

-   :fontawesome-solid-shapes: **Multiple Algorithms**
    ---
    Supports BPE, WordPiece, and Unigram

-   :material-cog-transfer: **Production Ready**
    ---
    Battle-tested and ready for deployment

</div>

## Overview

MyTokenizer is a high-performance, production-ready tokenizer that supports multiple tokenization algorithms including Byte Pair Encoding (BPE), WordPiece, and Unigram. Designed with both simplicity and performance in mind, it's perfect for natural language processing tasks where tokenization speed and accuracy are critical.

## Key Features

- **Multiple Tokenization Algorithms**: Choose between BPE, WordPiece, or Unigram based on your needs
- **Blazing Fast**: Optimized implementation for maximum performance
- **Memory Efficient**: Smart memory management for large datasets
- **Easy to Use**: Simple, intuitive API with sensible defaults
- **Production Ready**: Thoroughly tested and production-hardened
- **Extensible**: Easy to extend with custom tokenization logic

## Quick Example

```python
from tokenizer import Tokenizer

# Initialize with default settings
tokenizer = Tokenizer()

# Tokenize some text
tokens = tokenizer.tokenize("Hello, world! This is MyTokenizer in action.")
print(tokens)
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'My', '##Token', '##izer', 'in', 'action', '.']
```

## Getting Started

1. [Installation](installation.md) - Install MyTokenizer
2. [Quickstart](getting-started/quickstart.md) - Get up and running in minutes
3. [User Guide](guide/basic-usage.md) - Learn how to use all features
4. [API Reference](api/tokenizer.md) - Detailed API documentation

## Community

- **GitHub Issues**: Found a bug or have a feature request? [Open an issue](https://github.com/pranav271103/MyTokenizer/issues)
## Support and Community

If you need help or have questions, please [open an issue](https://github.com/pranav271103/MyTokenizer/issues). For commercial support, please contact [pranav.singh01010101@gmail.com](mailto:pranav.singh01010101@gmail.com).

## License and Contributing

This project is licensed under the MIT License. The source code is available on [GitHub](https://github.com/pranav271103/MyTokenizer). We welcome contributions from the community!

## Citation

If you use MyTokenizer in your research, please consider citing it:

```bibtex
@software{MyTokenizer,
  author = {Pranav Singh, Raman Mendiratta},
  title = {MyTokenizer: A High-Performance Tokenizer for NLP},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pranav271103/MyTokenizer}}
}