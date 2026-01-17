#!/usr/bin/env python3
"""
Test LiteLLM + Langfuse integration with OpenHands SDK.

This script verifies that:
1. LiteLLM is properly configured with Langfuse OTEL callback
2. OpenHands LLM calls are logged to Langfuse
3. Z.ai API works with the configuration
"""

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

print(f"✅ Environment loaded via python-dotenv")
print(f"  ZAI_BASE_URL: {os.getenv('ZAI_BASE_URL')}")
print(f"  LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")

# Enable Langfuse OTEL integration for LiteLLM
import litellm
litellm.callbacks = ["langfuse_otel"]
print(f"\n✅ LiteLLM callbacks configured: {litellm.callbacks}")

# Now test OpenHands SDK
from pydantic import SecretStr
from openhands.sdk import LLM

print("\n" + "="*60)
print("Testing OpenHands SDK with Z.ai + Langfuse")
print("="*60)

# Create LLM instance
llm = LLM(
    model="glm-4.7",
    api_key=SecretStr(os.getenv("ZAI_API_KEY")),
    base_url=os.getenv("ZAI_BASE_URL"),
    usage_id="test_langfuse_integration",
)

print(f"\n✅ LLM instance created")
print(f"  Model: {llm.model}")
print(f"  Base URL: {llm.base_url}")

# Test a simple completion
print("\n" + "-"*60)
print("Testing completion()...")
print("-"*60)

try:
    # Use responses() instead of completion() for now (streaming API)
    response_stream = llm.responses(
        messages=[
            {"role": "user", "content": "Say 'Hello from Z.ai via OpenHands with Langfuse!'"}
        ],
        metadata={
            "session_id": "test_session_001",
            "tags": ["test", "langfuse-integration", "zai"],
        }
    )
    
    # Collect the full response
    response_content = ""
    for chunk in response_stream:
        if chunk.content:
            response_content += chunk.content[0].text
    
    print(f"\n✅ Response successful!")
    print(f"  Response: {response_content}")
    
    # Check metrics
    metrics = llm.metrics
    print(f"\n📊 Metrics:")
    print(f"  Total cost: ${metrics.accumulated_cost:.6f}")
    print(f"  Total tokens: {metrics.accumulated_total_tokens}")
    
except Exception as e:
    print(f"\n❌ Error during completion:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete! Check Langfuse at http://localhost:3044")
print("="*60)
