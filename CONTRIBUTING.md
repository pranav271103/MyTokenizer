# Contributing to MyTokenizer

Thank you for your interest in contributing to MyTokenizer! We welcome all contributions, whether they're bug reports, feature requests, documentation improvements, or code contributions. This project is maintained by [Pranav Singh](mailto:pranav.singh01010101@gmail.com) and the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Review Process](#code-review-process)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as [GitHub issues](https://github.com/pranav271103/MyTokenizer/issues). Before creating a new issue, please search to see if a similar issue already exists. When creating a bug report, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected vs. actual behavior
4. Environment details (Python version, OS, etc.)
5. Any relevant logs or error messages

### Suggesting Enhancements

We welcome enhancement suggestions. Please open an issue with:

1. A clear, descriptive title
2. A detailed description of the enhancement
3. Use cases and examples
4. Any alternative solutions or workarounds

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you've added code, add tests.
3. If you've changed APIs, update the documentation.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/NLP.git
   cd NLP
   ```
3. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install  # Install git hooks
   ```
5. Create a new branch:
   ```bash
   git checkout -b feat/your-feature-name  # or fix/your-bug-fix
   ```
6. Make your changes following the style guidelines
7. Run tests and verify your changes:
   ```bash
   pytest  # Run all tests
   mypy .  # Type checking
   black . --check  # Code formatting
   isort . --check-only  # Import sorting
   ```
8. Commit your changes with a descriptive message:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
9. Push to the branch:
   ```bash
   git push origin feat/your-feature-name
   ```
10. Open a pull request against the `main` branch

## Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for all functions and methods
- Write docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep lines under 88 characters (Black's default)
- Use absolute imports
- Use single quotes for strings unless the string contains a single quote
- Document all public APIs with docstrings
- Add type hints to all function signatures
- Include example usage in docstrings when appropriate
- Keep functions small and focused on a single task

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. They will run automatically on commit:

- Black code formatting
- isort import sorting
- flake8 linting
- mypy type checking

## Testing

- Write tests for all new functionality
- Test both success and error cases
- Use descriptive test function names that describe the behavior being tested
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Run tests with:
  ```bash
  pytest  # Run all tests
  pytest tests/test_module.py  # Run specific test file
  pytest tests/test_module.py::test_function  # Run specific test
  pytest -v  # Verbose output
  pytest --cov=NLP --cov-report=term-missing  # Coverage report
  ```
- Maintain at least 80% test coverage
- Test on multiple Python versions (3.8+)
- Ensure all tests pass before submitting a PR

## Documentation

We use Sphinx for documentation. To build the docs:

```bash
cd docs
make html
```

Documentation should be updated for any new features or changes to existing functionality.

## Code Review Process

1. Ensure all CI checks pass (tests, linting, type checking)
2. Ensure code meets style guidelines
3. Request review from a maintainer
4. Address any feedback
5. Get approval from at least one maintainer
6. Squash and merge into main

## Release Process (for maintainers)

1. Update version in `NLP/__version__.py` following [Semantic Versioning](https://semver.org/)
2. Update `CHANGELOG.md` with changes
3. Create a release tag:
   ```bash
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   git push origin vX.Y.Z
   ```
4. Publish to PyPI:
   ```bash
   rm -rf dist/*
   python -m build
   twine upload dist/*
   ```
5. Create a GitHub release with release notes

## Need Help?

If you need help or have questions, please open an issue or contact [Pranav Singh](mailto:pranav.singh01010101@gmail.com).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
