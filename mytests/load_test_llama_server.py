#!/usr/bin/env python3
"""
Load testing script for llama.cpp server running Qwen3.5-27B model.
Server is configured to run on http://localhost:8888
"""

import sys
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import requests


@dataclass
class TestResult:
    """Result of a single request."""
    success: bool
    latency_ms: float
    tokens_generated: int = 0
    error_message: str = ""


@dataclass
class LoadTestStats:
    """Statistics from a load test run."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)

    @property
    def min_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return min(self.latencies)

    @property
    def max_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return max(self.latencies)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def requests_per_second(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_requests / self.total_time

    @property
    def tokens_per_second(self) -> float:
        if self.total_time == 0:
            return 0.0
        return sum(self.tokens) / self.total_time

    def print_report(self, test_name: str):
        """Print a formatted report of the test results."""
        print(f"\n{'=' * 60}")
        print(f"Load Test Results: {test_name}")
        print(f"{'=' * 60}")
        print(f"Total Requests:     {self.total_requests}")
        print(f"Successful:         {self.successful_requests} ({self.success_rate:.1f}%)")
        print(f"Failed:             {self.failed_requests}")
        print(f"\nLatency Statistics:")
        print(f"  Average:          {self.avg_latency_ms:.2f} ms")
        print(f"  Min:              {self.min_latency_ms:.2f} ms")
        print(f"  Max:              {self.max_latency_ms:.2f} ms")
        print(f"  P50 (Median):     {self.p50_latency_ms:.2f} ms")
        print(f"  P95:              {self.p95_latency_ms:.2f} ms")
        print(f"  P99:              {self.p99_latency_ms:.2f} ms")
        print(f"\nThroughput:")
        print(f"  Requests/sec:     {self.requests_per_second:.2f}")
        print(f"  Tokens/sec:       {self.tokens_per_second:.2f}")
        print(f"  Total Time:       {self.total_time:.2f} s")
        print(f"{'=' * 60}\n")


class LlamaLoadTester:
    """Load testing client for llama.cpp server."""

    def __init__(self, host: str = "localhost", port: int = 8888):
        self.base_url = f"http://{host}:{port}"
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.health_url = f"{self.base_url}/health"
        self.session = requests.Session()

    def check_health(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            response = self.session.get(self.health_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def make_completion_request(
        self,
        prompt: str,
        model: str = "mymodel",
        max_tokens: int = 100,
        temperature: float = 0.7,
        timeout: int = 120,
        print_errors: bool = False
    ) -> TestResult:
        """Make a single completion request and return timing results."""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        start_time = time.time()
        try:
            response = self.session.post(
                self.completions_url,
                json=payload,
                timeout=timeout
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return TestResult(success=True, latency_ms=latency_ms, tokens_generated=tokens)
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                if print_errors:
                    print(f"  Error: {error_msg}")
                return TestResult(
                    success=False,
                    latency_ms=latency_ms,
                    error_message=error_msg
                )
        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            if print_errors:
                print(f"  Error: {error_msg}")
            return TestResult(success=False, latency_ms=latency_ms, error_message=error_msg)

    def make_chat_request(
        self,
        messages: list[dict],
        model: str = "mymodel",
        max_tokens: int = 100,
        temperature: float = 0.7,
        timeout: int = 120
    ) -> TestResult:
        """Make a single chat completion request and return timing results."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        start_time = time.time()
        try:
            response = self.session.post(
                self.chat_url,
                json=payload,
                timeout=timeout
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return TestResult(success=True, latency_ms=latency_ms, tokens_generated=tokens)
            else:
                return TestResult(
                    success=False,
                    latency_ms=latency_ms,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            return TestResult(success=False, latency_ms=latency_ms, error_message=str(e))

    def run_concurrent_test(
        self,
        request_func,
        num_requests: int,
        max_concurrent: int,
        *args,
        **kwargs
    ) -> LoadTestStats:
        """Run concurrent requests and collect statistics."""
        stats = LoadTestStats()
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(request_func, *args, **kwargs) for _ in range(num_requests)]

            for future in as_completed(futures):
                result = future.result()
                stats.total_requests += 1

                if result.success:
                    stats.successful_requests += 1
                    stats.latencies.append(result.latency_ms)
                    stats.tokens.append(result.tokens_generated)
                else:
                    stats.failed_requests += 1

        stats.total_time = time.time() - start_time
        return stats

    def run_load_test(
        self,
        prompt: str,
        num_requests: int = 10,
        max_concurrent: int = 2,
        max_tokens: int = 100,
        model: str = "mymodel",
        test_type: str = "completion"
    ) -> LoadTestStats:
        """Run a load test with the specified parameters."""
        if test_type == "chat":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            return self.run_concurrent_test(
                self.make_chat_request,
                num_requests,
                max_concurrent,
                messages,
                model=model,
                max_tokens=max_tokens
            )
        else:
            return self.run_concurrent_test(
                self.make_completion_request,
                num_requests,
                max_concurrent,
                prompt,
                model=model,
                max_tokens=max_tokens
            )


def generate_large_context(num_tokens: int = 200000) -> str:
    """Generate a large context close to the specified token count."""
    # Average English token is ~4 characters
    target_chars = num_tokens * 4

    base_block = """
    # Technical Documentation: Python Programming Language

    ## Chapter 1: Introduction to Python
    Python is a high-level, interpreted, general-purpose programming language.
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python statements commonly end with a newline rather than a semicolon.
    It has fewer syntactical conventions than languages like C or Java.

    ## Chapter 2: Data Types and Structures
    Python supports various built-in data types including integers, floats, strings,
    booleans, lists, tuples, dictionaries, and sets. Each type has specific operations
    and methods that can be performed on it.

    ## Chapter 3: Control Flow
    Control flow statements in Python include if-elif-else for conditional execution,
    for and while loops for iteration, and break/continue for loop control.
    Python also supports list comprehensions and generator expressions for concise iteration.

    ## Chapter 4: Functions and Modules
    Functions are defined using the def keyword. Python supports default arguments,
    keyword arguments, and variable-length arguments (*args and **kwargs).
    Modules allow code organization and reuse through the import statement.

    ## Chapter 5: Object-Oriented Programming
    Python is an object-oriented language that supports classes, inheritance,
    polymorphism, and encapsulation. Everything in Python is an object.
    Classes are defined using the class keyword, and methods are functions defined within classes.

    ## Chapter 6: Exception Handling
    Python uses try-except blocks for exception handling. Common exceptions include
    ValueError, TypeError, KeyError, IndexError, and FileNotFoundError.
    Custom exceptions can be created by inheriting from the Exception class.

    ## Chapter 7: File I/O
    File operations in Python use the open() function which returns a file object.
    Files can be read, written, and appended using appropriate modes ('r', 'w', 'a').
    The with statement ensures proper file closing through context managers.

    ## Chapter 8: Standard Library
    Python's standard library includes modules for regular expressions (re),
    date and time handling (datetime), mathematical operations (math),
    random number generation (random), and many more utilities.

    ## Chapter 9: Virtual Environments
    Virtual environments allow isolated Python environments with separate package installations.
    They are created using venv or virtualenv modules and help manage project dependencies.

    ## Chapter 10: Package Management
    pip is the package installer for Python, used to install and manage packages from PyPI.
    Requirements files (requirements.txt) help document and reproduce project dependencies.
    """

    # Calculate repetitions needed
    block_size = len(base_block)
    repetitions = target_chars // block_size

    large_context = base_block * repetitions
    prompt = f"{large_context}\n\nBased on the documentation above, summarize the key topics covered in Python programming."

    return prompt


def run_large_context_test(
    num_requests: int = 2,
    max_concurrent: int = 2,
    max_tokens: int = 100,
    context_size: int = 100000,
    timeout: int = 600
):
    """Run a large context load test with parallel requests.

    Args:
        num_requests: Number of requests to make
        max_concurrent: Maximum concurrent requests
        max_tokens: Maximum output tokens per request
        context_size: Target context size in tokens (default 200k)
        timeout: Request timeout in seconds (default 600 = 10 minutes)
    """
    tester = LlamaLoadTester()

    if not tester.check_health():
        print("Server is not responding. Please start the server first.")
        return

    print("\n" + "=" * 60)
    print("Large Context Load Test")
    print("=" * 60)

    # Generate large context
    print(f"\nGenerating context (~{context_size:,} tokens)...")
    large_prompt = generate_large_context(context_size)

    print(f"Context size: {len(large_prompt):,} characters (~{len(large_prompt)//4:,} tokens)")
    print(f"Requests: {num_requests}")
    print(f"Max Concurrent: {max_concurrent}")
    print(f"Max Output Tokens: {max_tokens}")
    print(f"Timeout: {timeout} seconds")
    print("\nStarting load test...")

    # Make requests with extended timeout for large context
    stats = LoadTestStats()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(
                tester.make_completion_request,
                large_prompt,
                model="mymodel",
                max_tokens=max_tokens,
                timeout=timeout,
                print_errors=True
            )
            for _ in range(num_requests)
        ]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            stats.total_requests += 1
            if result.success:
                stats.successful_requests += 1
                stats.latencies.append(result.latency_ms)
                stats.tokens.append(result.tokens_generated)
                print(f"  Request {i+1}: SUCCESS - {result.latency_ms:.2f}ms, {result.tokens_generated} tokens")
            else:
                stats.failed_requests += 1
                print(f"  Request {i+1}: FAILED - {result.error_message[:100]}")

    stats.total_time = time.time() - start_time
    stats.print_report(f"Large Context ({num_requests} req, {max_concurrent} concurrent, ~{context_size//1000}k context)")


def run_custom_test(
    prompt: str,
    num_requests: int,
    max_concurrent: int,
    max_tokens: int = 100,
    model: str = "mymodel",
    test_type: str = "completion"
):
    """Run a custom load test with user-specified parameters."""
    tester = LlamaLoadTester()

    if not tester.check_health():
        print("Server is not responding. Please start the server first.")
        return

    print(f"\nRunning custom load test:")
    print(f"  Prompt: {prompt[:50]}...")
    print(f"  Requests: {num_requests}")
    print(f"  Max Concurrent: {max_concurrent}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Test Type: {test_type}")
    print()

    stats = tester.run_load_test(
        prompt,
        num_requests=num_requests,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        model=model,
        test_type=test_type
    )
    stats.print_report("Custom Test")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "large-context":
            # Large context test: python load_test.py large-context [requests] [concurrent] [max_tokens] [context_size] [timeout]
            num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 2
            max_concurrent = int(sys.argv[3]) if len(sys.argv) > 3 else 2
            max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 100
            context_size = int(sys.argv[5]) if len(sys.argv) > 5 else 100000  # Default to 100k tokens
            timeout = int(sys.argv[6]) if len(sys.argv) > 6 else 600  # 10 minutes timeout
            run_large_context_test(num_requests, max_concurrent, max_tokens, context_size, timeout)
        else:
            # Custom test mode: python load_test.py <prompt> <requests> <concurrent> [max_tokens]
            prompt = cmd
            num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            max_concurrent = int(sys.argv[3]) if len(sys.argv) > 3 else 2
            max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 100
            run_custom_test(prompt, num_requests, max_concurrent, max_tokens)
    else:
        print("Usage:")
        print("  python load_test_llama_server.py                    # Run all tests")
        print("  python load_test_llama_server.py large-context       # Run large context test (200k)")
        print("  python load_test_llama_server.py large-context <req> <conc> <tokens> <ctx_size> <timeout>")
        print("  python load_test_llama_server.py '<prompt>' <req> <conc> [tokens]  # Custom test")
