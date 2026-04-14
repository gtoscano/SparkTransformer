#!/usr/bin/env python3
"""
vLLM Container Test and Benchmark Script
Tests if vLLM is running and measures tokens/second performance
"""

import requests
import time
import json
from typing import Dict, Any

# Configuration — model is auto-detected from the running server
BASE_URL = "http://localhost:8355"
VLLM_URL = f"{BASE_URL}/v1/completions"
VLLM_HEALTH_URL = f"{BASE_URL}/health"
VLLM_MODELS_URL = f"{BASE_URL}/v1/models"
MODEL = None  # auto-detected

# 1000-word story prompt
STORY_PROMPT = """Write a detailed science fiction story about a space explorer discovering an ancient alien civilization on a distant planet. The story should be approximately 1000 words and include:
- A vivid description of the planet's landscape
- The explorer's emotional journey
- Details about the alien civilization's architecture and technology
- A mysterious discovery that changes everything
- A suspenseful conclusion

Begin the story now:"""


def check_vllm_status() -> bool:
    """Check if vLLM server is running and healthy"""
    print("=" * 60)
    print("STEP 1: Checking vLLM Server Status")
    print("=" * 60)

    try:
        response = requests.get(VLLM_HEALTH_URL, timeout=5)
        if response.status_code == 200:
            print("✓ Server is running and healthy")
            return True
    except Exception:
        pass

    # Fallback: try /v1/models (TRT-LLM may not have /health)
    try:
        response = requests.get(VLLM_MODELS_URL, timeout=5)
        if response.status_code == 200:
            print("✓ Server is running (/v1/models responded)")
            return True
    except Exception:
        pass

    print(f"✗ Cannot connect to server at {BASE_URL}")
    print("  Make sure the docker container is running:")
    print("  docker ps")
    return False


def get_model_info() -> Dict[str, Any]:
    """Get information about loaded model and auto-detect MODEL name"""
    global MODEL
    print("\n" + "=" * 60)
    print("STEP 2: Getting Model Information")
    print("=" * 60)

    try:
        response = requests.get(VLLM_MODELS_URL, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            if models_data.get('data'):
                model_info = models_data['data'][0]
                MODEL = model_info.get('id', None)
                print(f"✓ Model loaded: {MODEL}")
                print(f"  API endpoint: {BASE_URL}/v1/chat/completions")
                print(f"  Model name:   {MODEL}")
                return model_info
            else:
                print("✗ No model information available")
        else:
            print(f"✗ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"✗ Error getting model info: {e}")

    return {}


def benchmark_tokens_per_second(max_tokens: int = 1500) -> Dict[str, Any]:
    """
    Benchmark the vLLM server with a story generation task

    Args:
        max_tokens: Maximum tokens to generate (should be enough for ~1000 words)

    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "=" * 60)
    print("STEP 3: Benchmarking Tokens Per Second")
    print("=" * 60)
    print(f"Generating story (max {max_tokens} tokens)...")
    print("This may take a minute...\n")

    payload = {
        "model": MODEL,
        "prompt": STORY_PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(
            VLLM_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()

            # Extract metrics
            total_time = end_time - start_time
            generated_text = result['choices'][0]['text']

            # Get token counts from response if available
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

            # Calculate metrics
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            word_count = len(generated_text.split())

            # Display results
            print("=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(f"✓ Generation completed successfully!\n")
            print(f"Timing:")
            print(f"  Total time:           {total_time:.2f} seconds")
            print(f"\nTokens:")
            print(f"  Prompt tokens:        {prompt_tokens}")
            print(f"  Completion tokens:    {completion_tokens}")
            print(f"  Total tokens:         {total_tokens}")
            print(f"\nPerformance:")
            print(f"  Tokens per second:    {tokens_per_second:.2f} tok/s")
            print(f"  Words generated:      {word_count}")
            print(f"  Words per second:     {word_count/total_time:.2f} words/s")

            print("\n" + "=" * 60)
            print("GENERATED STORY (first 500 characters)")
            print("=" * 60)
            print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
            print("\n" + "=" * 60)

            return {
                'success': True,
                'total_time': total_time,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'tokens_per_second': tokens_per_second,
                'word_count': word_count,
                'generated_text': generated_text
            }
        else:
            print(f"✗ Error: Server returned status {response.status_code}")
            print(f"Response: {response.text}")
            return {'success': False, 'error': response.text}

    except requests.exceptions.Timeout:
        print("✗ Request timed out. The model might be taking too long to respond.")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"✗ Error during benchmark: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("vLLM Performance Benchmark Tool")
    print("=" * 60)
    print()

    # Step 1: Check if server is running
    if not check_vllm_status():
        print("\n❌ Cannot proceed - vLLM server is not accessible")
        print("\nTroubleshooting steps:")
        print("1. Check if container is running: docker ps")
        print(f"2. Ensure {BASE_URL} is reachable")
        return

    # Step 2: Get model information (auto-detects MODEL)
    get_model_info()
    if not MODEL:
        print("\n❌ Cannot proceed - no model detected")
        return

    # Step 3: Run benchmark
    results = benchmark_tokens_per_second()

    # Summary
    if results.get('success'):
        print("\n✅ Benchmark completed successfully!")
        print(f"\n🚀 Your vLLM server is generating at {results['tokens_per_second']:.2f} tokens/second")
    else:
        print("\n❌ Benchmark failed")
        print("Check the error messages above for details")


if __name__ == "__main__":
    main()
