"""
Benchmark script to evaluate the custom tokenizer against existing tokenizers.
"""

import os
import time
import logging
from tokenizer import Tokenizer, TokenizerTrainer
from tokenizer.evaluation import TokenizerEvaluator
from tokenizer.benchmark_scenarios import BenchmarkScenarios
from tokenizer.memory_profiler import MemoryProfiler
from tokenizer.memory_utils import MemoryManager
import numpy as np
from tokenizer.vocab import Vocabulary
from typing import List, Dict, Any
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_data(scenario: str = "mixed_lengths") -> List[str]:
    """Load test data for benchmarking.

    Args:
        scenario: Scenario to use ('mixed_lengths', 'long_texts', 'short_texts', 'repeated_patterns', 'multilingual', 'special_chars')

    Returns:
        List of test texts
    """
    scenarios = BenchmarkScenarios(num_samples=1000)
    all_scenarios = scenarios.generate_all_scenarios()

    if scenario not in all_scenarios:
        raise ValueError(
            f"Unknown scenario: {scenario}. Available scenarios: {list(all_scenarios.keys())}"
        )

    return all_scenarios[scenario]


def evaluate_speed(test_texts: List[str], scenario: str) -> Dict[str, Any]:
    """Evaluate tokenization speed with memory-aware batching and profiling.

    Args:
        test_texts: Texts to process
        scenario: Scenario name for logging

    Returns:
        Dictionary of performance metrics
    """
    tokenizer = Tokenizer()

    # Initialize memory profiler
    profiler = MemoryProfiler()
    profiler.start()

    # Warm up cache
    warmup_texts = test_texts[:100]
    for text in warmup_texts:
        tokenizer.tokenize(text)

    # Find optimal batch size
    logger.info(f"Finding optimal batch size for {scenario}...")
    optimal_batch_size = tokenizer.find_optimal_batch_size(test_texts)
    logger.info(f"Using optimal batch size: {optimal_batch_size} for {scenario}")

    # Take initial memory snapshot
    profiler.snapshot(f"initial_{scenario}")

    # Measure single-threaded tokenization speed
    start_time = time.time()
    for text in test_texts:
        tokenizer.tokenize(text)
    end_time = time.time()

    single_thread_time = end_time - start_time
    single_thread_tokens_per_second = len(test_texts) / single_thread_time

    # Take snapshot after single-threaded processing
    profiler.snapshot(f"single_thread_{scenario}")

    # Measure memory-aware parallel tokenization
    start_time = time.time()
    results = tokenizer.tokenize_batch(
        test_texts,
        batch_size=optimal_batch_size,
        max_memory=None,  # Let it use automatic memory management
    )
    end_time = time.time()

    parallel_time = end_time - start_time
    parallel_tokens_per_second = len(test_texts) / parallel_time

    # Take snapshot after parallel processing
    profiler.snapshot(f"parallel_{scenario}")

    # Measure memory-aware streaming tokenization
    start_time = time.time()
    results = list(
        tokenizer.tokenize_stream(
            (text for text in test_texts), batch_size=optimal_batch_size
        )
    )
    end_time = time.time()

    streaming_time = end_time - start_time
    streaming_tokens_per_second = len(test_texts) / streaming_time

    # Take final snapshot
    profiler.snapshot(f"final_{scenario}")

    # Get memory statistics
    memory_stats = {
        "available_memory": MemoryManager.get_available_memory(),
        "recommended_batch_size": MemoryManager.get_recommended_batch_size(
            test_texts[:1000]
        ),
        "profiler_stats": profiler.get_memory_stats(),
    }

    # Stop profiling
    memory_stats.update(profiler.stop())

    return {
        "scenario": scenario,
        "single_thread": {
            "avg_time": single_thread_time / len(test_texts),
            "tokens_per_second": single_thread_tokens_per_second,
        },
        "parallel": {
            "avg_time": parallel_time / len(test_texts),
            "tokens_per_second": parallel_tokens_per_second,
        },
        "streaming": {
            "avg_time": streaming_time / len(test_texts),
            "tokens_per_second": streaming_tokens_per_second,
        },
        "optimal_batch_size": optimal_batch_size,
        "memory_stats": memory_stats,
    }


def evaluate_tokenizer() -> Dict[str, Any]:
    """Evaluate tokenizer performance across multiple scenarios."""
    # Initialize tokenizer
    trainer = TokenizerTrainer(vocab_size=1000, min_frequency=1, lowercase=True)

    # Train tokenizer (using example text)
    logger.info("Training tokenizer...")
    tokenizer = trainer.train(files=["example.txt"], algorithm="bpe", num_workers=1)

    # Initialize evaluator
    evaluator = TokenizerEvaluator(tokenizer)

    # Load test data
    scenarios = BenchmarkScenarios(num_samples=1000)
    all_scenarios = scenarios.generate_all_scenarios()

    results = {}

    for scenario_name, scenario_texts in all_scenarios.items():
        logger.info(f"Evaluating scenario: {scenario_name}")

        # Evaluate speed for this scenario
        speed_results = evaluate_speed(scenario_texts, scenario_name)

        # Run other evaluations using TokenizerEvaluator
        coverage_metrics = evaluator.evaluate_coverage(scenario_texts)
        speed_metrics = evaluator.evaluate_speed(scenario_texts)
        consistency_metrics = evaluator.evaluate_consistency(scenario_texts)
        subword_metrics = evaluator.evaluate_subword_quality(scenario_texts)

        # Store results for this scenario
        results[scenario_name] = {
            "speed": speed_results,
            "coverage": coverage_metrics,
            "speed_eval": speed_metrics,
            "consistency": consistency_metrics,
            "subword_quality": subword_metrics,
        }

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


def print_evaluation_summary(results: Dict[str, Any]):
    """Print evaluation summary."""
    print("\nEvaluation Summary:\n")

    # Print results for each scenario
    for scenario_name, scenario_results in results.items():
        print(f"=== {scenario_name.upper()} ===")

        if "coverage" in scenario_results:
            print("Vocabulary Coverage:")
            print(f"  Coverage: {scenario_results['coverage']['vocab_coverage']:.2f}%")
            print(
                f"  Unknown Token Rate: {scenario_results['coverage']['unknown_token_rate']:.2f}%"
            )
            print(
                f"  Avg Token Length: {scenario_results['coverage']['avg_token_length']:.2f}"
            )

        if "speed" in scenario_results:
            print("Tokenization Speed:")
            print(
                f"  Single Thread Time: {scenario_results['speed']['single_thread']['avg_time']:.4f} seconds"
            )
            print(
                f"  Parallel Time: {scenario_results['speed']['parallel']['avg_time']:.4f} seconds"
            )
            print(
                f"  Streaming Time: {scenario_results['speed']['streaming']['avg_time']:.4f} seconds"
            )
            print(
                f"  Single Thread Tokens/sec: {scenario_results['speed']['single_thread']['tokens_per_second']:.2f}"
            )
            print(
                f"  Parallel Tokens/sec: {scenario_results['speed']['parallel']['tokens_per_second']:.2f}"
            )
            print(
                f"  Streaming Tokens/sec: {scenario_results['speed']['streaming']['tokens_per_second']:.2f}"
            )
            print(
                f"  Optimal Batch Size: {scenario_results['speed']['optimal_batch_size']}"
            )

        if "consistency" in scenario_results:
            print("Tokenization Consistency:")
            print(
                f"  Token Variance: {scenario_results['consistency']['token_variance']:.2f}"
            )
            print(
                f"  Unique Tokens Ratio: {scenario_results['consistency']['unique_tokens_ratio']:.2f}"
            )

        if "subword_quality" in scenario_results:
            print("Subword Quality:")
            print(
                f"  Avg Subwords per Word: {scenario_results['subword_quality']['avg_subwords_per_word']:.2f}"
            )
            print(
                f"  Subword Repetition Rate: {scenario_results['subword_quality']['subword_repetition_rate']:.2f}"
            )

        print()


def main():
    """Main function to run benchmarks."""
    # Create example text file
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(load_test_data()))

    try:
        # Run evaluations
        metrics = evaluate_tokenizer()

        # Save results
        evaluator = TokenizerEvaluator(None)  # Just for saving
        evaluator.save_metrics(metrics, "benchmark_results.json")

        # Print summary
        print_evaluation_summary(metrics)
    finally:
        # Clean up
        if os.path.exists("example.txt"):
            os.remove("example.txt")


if __name__ == "__main__":
    main()
