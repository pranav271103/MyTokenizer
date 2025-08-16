# Changelog

All notable changes to the MyTokenizer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of the tokenizer with BPE, WordPiece, and Unigram support
- Comprehensive test suite for core functionality
- CLI interface for training and tokenization
- Documentation including README, CONTRIBUTING, and CODE_OF_CONDUCT
- GitHub Actions workflow for CI/CD

### Changed
- Updated project metadata and configuration files
- Improved error handling and input validation
- Enhanced documentation and code examples

### Fixed
- Resolved cache eviction bug in Vocabulary.tokenize
- Fixed packaging and installation issues
- Addressed various edge cases in tokenization

## [0.1.0] - 2025-08-16

### Added
- Initial release of MyTokenizer
- Core tokenization functionality
- Support for BPE, WordPiece, and Unigram algorithms
- Basic training and inference pipelines

[Unreleased]: https://github.com/pranav271103/MyTokenizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pranav271103/MyTokenizer/releases/tag/v0.1.0
