#!/usr/bin/env python3
"""
MyTokenizer Demo

A simple interactive demo for the Ultra-Tokenizer package.
"""

import argparse
import os
import sys
from typing import List, Optional

from tokenizer import Tokenizer, TokenizerTrainer


def train_demo_tokenizer() -> Tokenizer:
    """Train a simple tokenizer for demo purposes."""
    print("\nTraining a simple tokenizer for demonstration...")

    # Sample training data
    sample_texts = [
        "This is a sample sentence for the demo tokenizer.",
        "The tokenizer supports multiple languages and special characters.",
        "You can tokenize any text input after this initial training.",
        "The tokenizer uses BPE algorithm by default.",
        "Try entering your own text to see it in action!",
    ]

    # Create a temporary file for training
    temp_file = "demo_training_data.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_texts))

    # Initialize and train the tokenizer
    trainer = TokenizerTrainer(
        vocab_size=1000, min_frequency=1, lowercase=True, strip_accents=True
    )

    tokenizer = trainer.train(files=[temp_file], algorithm="bpe", num_workers=1)

    # Clean up
    try:
        os.remove(temp_file)
    except OSError:
        pass

    return tokenizer


def interactive_demo():
    """Run an interactive demo of the tokenizer."""
    print("=" * 60)
    print("Ultra-Tokenizer - Interactive Demo")
    print("=" * 60)
    print("\nThis demo will train a simple tokenizer and allow you to test it.")

    # Train a simple tokenizer
    tokenizer = train_demo_tokenizer()

    print("\n" + "=" * 60)
    print("Tokenization Demo")
    print("Type 'exit' or press Ctrl+C to quit")
    print("=" * 60)

    try:
        while True:
            # Get user input
            try:
                text = input("\nEnter text to tokenize: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting demo...")
                break

            if text.lower() in ("exit", "quit"):
                print("Exiting demo...")
                break

            if not text:
                print("Please enter some text to tokenize.")
                continue

            # Tokenize the input
            try:
                tokens = tokenizer.tokenize(text)
                print("\nTokens:")
                print("-" * 60)
                # Display tokens in a more readable format
                line = ""
                for i, token in enumerate(tokens, 1):
                    token_repr = f"[{token}] "
                    if len(line) + len(token_repr) > 60:
                        print(line)
                        line = token_repr
                    else:
                        line += token_repr
                if line:
                    print(line)
                print("-" * 60)
                print(f"Token count: {len(tokens)}")

                # Show character-level tokens as well
                print("\nCharacter-level tokens:")
                print("-" * 60)
                chars = list(text)
                print(" ".join(f"[{c}]" if c.strip() else "[ ]" for c in chars))
                print("-" * 60)
                print(f"Character count: {len(chars)}")

            except Exception as e:
                print(f"Error during tokenization: {e}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nThank you for trying Ultra-Tokenizer!")


def main():
    parser = argparse.ArgumentParser(description="Ultra-Tokenizer Demo")
    parser.add_argument(
        "--text",
        type=str,
        help="Text to tokenize (if not provided, runs in interactive mode)",
        default=None,
    )

    args = parser.parse_args()

    if args.text:
        # Non-interactive mode
        tokenizer = train_demo_tokenizer()
        print("\nInput text:")
        print(f'"{args.text}"')
        print("\nTokens:")
        print("-" * 60)
        tokens = tokenizer.tokenize(args.text)
        # Display tokens in a more readable format
        line = ""
        for i, token in enumerate(tokens, 1):
            token_repr = f"[{token}] "
            if len(line) + len(token_repr) > 60:
                print(line)
                line = token_repr
            else:
                line += token_repr
        if line:
            print(line)
        print("-" * 60)
        print(f"Token count: {len(tokens)}")

        # Show character-level tokens as well
        print("\nCharacter-level tokens:")
        print("-" * 60)
        chars = list(args.text)
        print(" ".join(f"[{c}]" if c.strip() else "[ ]" for c in chars))
        print("-" * 60)
        print(f"Character count: {len(chars)}")
    else:
        # Interactive mode
        interactive_demo()


if __name__ == "__main__":
    main()
