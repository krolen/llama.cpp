#!/usr/bin/env python3
"""
Test script for llama.cpp server running Qwen3.5-27B model.
Server is configured to run on http://localhost:8888
"""

import sys
import requests
import json
import time
from typing import Optional


class LlamaServerTester:
    """Test client for llama.cpp server."""

    def __init__(self, host: str = "localhost", port: int = 8888):
        self.base_url = f"http://{host}:{port}"
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"
        self.health_url = f"{self.base_url}/health"

    def check_health(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return False

    def list_models(self) -> Optional[list]:
        """List available models on the server."""
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return None
        except requests.exceptions.RequestException as e:
            print(f"Failed to list models: {e}")
            return None

    def generate_completion(
        self,
        prompt: str,
        model: str = "mymodel",
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        stream: bool = False
    ) -> Optional[dict]:
        """Generate a completion using the completions API."""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        try:
            response = requests.post(
                self.completions_url,
                json=payload,
                timeout=120,
                stream=stream
            )

            if response.status_code == 200:
                if stream:
                    for line in response.iter_lines():
                        if line:
                            try:
                                # Handle SSE format: "data: {...}"
                                line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                                if line_str.startswith("data: "):
                                    json_str = line_str[6:]
                                    # Skip "[DONE]" messages
                                    if json_str == "[DONE]":
                                        continue
                                    data = json.loads(json_str)
                                    if "choices" in data:
                                        for choice in data["choices"]:
                                            if "text" in choice:
                                                print(choice["text"], end="", flush=True)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Failed to parse line: {line_str[:50]}... Error: {e}", file=sys.stderr)
                                continue
                    print()
                    return None
                return response.json()
            print(f"Completion failed with status {response.status_code}: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Completion request failed: {e}")
            return None

    def generate_chat_completion(
        self,
        messages: list[dict],
        model: str = "mymodel",
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        stream: bool = False
    ) -> Optional[dict]:
        """Generate a chat completion using the chat completions API."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120,
                stream=stream
            )

            if response.status_code == 200:
                if stream:
                    for line in response.iter_lines():
                        if line:
                            try:
                                # Handle SSE format: "data: {...}"
                                line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                                if line_str.startswith("data: "):
                                    json_str = line_str[6:]
                                    # Skip "[DONE]" messages
                                    if json_str == "[DONE]":
                                        continue
                                    data = json.loads(json_str)
                                    if "choices" in data:
                                        for choice in data["choices"]:
                                            if "delta" in choice and "content" in choice["delta"]:
                                                print(choice["delta"]["content"], end="", flush=True)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Failed to parse line: {line_str[:50]}... Error: {e}", file=sys.stderr)
                                continue
                    print()
                    return None
                return response.json()
            print(f"Chat completion failed with status {response.status_code}: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Chat completion request failed: {e}")
            return None


def run_tests():
    """Run a series of tests against the llama server."""
    tester = LlamaServerTester()

    print("=" * 60)
    print("Llama Server Test Suite")
    print("=" * 60)

    # Test 1: Health Check
    print("\n[TEST 1] Health Check")
    print("-" * 40)
    if tester.check_health():
        print("✓ Server is healthy and running")
    else:
        print("✗ Server is not responding")
        print("\nPlease start the server with: ./start-coding.sh")
        return

    # Test 2: List Models
    print("\n[TEST 2] List Available Models")
    print("-" * 40)
    models = tester.list_models()
    if models:
        for model in models:
            print(f"  - {model.get('id')}: {model.get('object')}")
    else:
        print("  No models found or failed to retrieve")

    # Test 3: Simple Completion
    print("\n[TEST 3] Simple Completion Test")
    print("-" * 40)
    prompt = "What is the capital of France?"
    print(f"Prompt: {prompt}")
    result = tester.generate_completion(prompt, max_tokens=50)
    if result:
        print(f"Response: {result['choices'][0]['text'].strip()}")
        print(f"Tokens used: {result['usage']['total_tokens']}")
    else:
        print("Failed to generate completion")

    # Test 4: Code Generation Test
    print("\n[TEST 4] Code Generation Test")
    print("-" * 40)
    code_prompt = "Write a Python function to calculate fibonacci numbers using memoization."
    print(f"Prompt: {code_prompt}")
    result = tester.generate_completion(code_prompt, max_tokens=256)
    if result:
        print(f"Response:\n{result['choices'][0]['text']}")
        print(f"Tokens used: {result['usage']['total_tokens']}")
    else:
        print("Failed to generate completion")

    # Test 5: Chat Completion Test
    print("\n[TEST 5] Chat Completion Test")
    print("-" * 40)
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain what a decorator is in Python."}
    ]
    print(f"User message: {messages[1]['content']}")
    result = tester.generate_chat_completion(messages, max_tokens=200)
    if result:
        print(f"Response:\n{result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']['total_tokens']}")
    else:
        print("Failed to generate chat completion")

    # Test 6: Streaming Completion Test
    print("\n[TEST 6] Streaming Completion Test")
    print("-" * 40)
    stream_prompt = "List three benefits of using version control."
    print(f"Prompt: {stream_prompt}")
    print("Response (streaming):")
    tester.generate_completion(stream_prompt, max_tokens=150, stream=True)

    # Test 7: Context Window Test
    print("\n[TEST 7] Context Window Test")
    print("-" * 40)
    long_context = """
    Here is a technical document about Python async programming:

    Python's asyncio library provides infrastructure for writing concurrent code using the await syntax.
    The key concepts include:

    1. Coroutines: Functions defined with async def that can be paused and resumed.
    2. Events Loop: The runtime system that executes coroutines.
    3. Tasks: Wrapper objects that schedule coroutines on the event loop.
    4. Futures: Objects representing the result of an asynchronous operation.

    To use asyncio effectively:
    - Use await to pause a coroutine until an awaitable completes
    - Use asyncio.gather() to run multiple coroutines concurrently
    - Use asyncio.create_task() to schedule a coroutine as a Task
    - Use async with for asynchronous context managers

    Common patterns include:
    - Reading multiple files concurrently
    - Making parallel HTTP requests
    - Processing data streams
    - Implementing network servers

    Best practices:
    - Avoid blocking operations in async code
    - Use asyncio.sleep() instead of time.sleep()
    - Properly handle exceptions with try/except
    - Use timeouts to prevent hanging operations
    """
    question = "Based on the above document, what are the four key concepts of asyncio?"
    prompt = f"{long_context}\n\nQuestion: {question}"
    print(f"Context length: {len(long_context)} characters")
    print(f"Question: {question}")
    result = tester.generate_completion(prompt, max_tokens=150)
    if result:
        print(f"Response: {result['choices'][0]['text'].strip()}")
    else:
        print("Failed to generate completion")

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
