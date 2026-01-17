import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Enable Langfuse OTEL integration for LiteLLM
import litellm
litellm.callbacks = ["langfuse_otel"]

print(f"✅ LiteLLM callbacks configured: {litellm.callbacks}")
print(f"  LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

llm = LLM(
    model=os.getenv("LLM_MODEL", "anthropic/glm-4.7"),  # Add anthropic/ prefix for LiteLLM
    api_key=os.getenv("ZAI_API_KEY"),
    base_url=os.getenv("ZAI_BASE_URL"),
)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ],
)

cwd = os.getcwd()
conversation = Conversation(agent=agent, workspace=cwd)

print("\n" + "="*60)
print("Running agent with Langfuse instrumentation...")
print("="*60)

conversation.send_message("Say 'Hello from Z.ai with Langfuse!' Keep it brief.")
conversation.run()

print("\n" + "="*60)
print("✅ Test complete! Check Langfuse at http://localhost:3044")
print("="*60)

# Show metrics
metrics = llm.metrics
print(f"\n📊 Metrics:")
print(f"  Total cost: ${metrics.accumulated_cost:.6f}")
print(f"  Input tokens: {metrics.accumulated_input_tokens}")
print(f"  Output tokens: {metrics.accumulated_output_tokens}")
