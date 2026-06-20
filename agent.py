from google.adk.agents import LlmAgent
import config
from state import RunState
from arena_tools import make_arena_tools
from helper_tools import make_helper_tools

def build_agent(state: RunState) -> LlmAgent:
    """Build the ADK LlmAgent with instructions and all standard + helper tools."""
    arena_tools = make_arena_tools(state)
    helper_tools = make_helper_tools(state)
    all_tools = arena_tools + helper_tools

    system_instruction = (
        "You are an autonomous Agent Arena Competitor. Your goal is to complete tasks in the Agent Arena, "
        "score >= 70, and level up.\n\n"
        "Workflow:\n"
        "1. First, register the agent using the `register_agent` tool if you haven't already. "
        f"Pass AGENT_NAME ({config.AGENT_NAME}) and AGENT_STACK ({config.AGENT_STACK}) from config.\n"
        "2. Fetch your current task using `get_tasks` tool with your AGENT_ID.\n"
        "3. Once you retrieve the task, read its description carefully. Determine what resources you need:\n"
        "   - Use `web_search` to look up current events, facts, definitions, or documentation.\n"
        "   - Use `calculate` for numeric equations, estimations, or straightforward arithmetic.\n"
        "   - Use `run_python` to write and execute python code. This is your most powerful tool! Use it to test your algorithms, "
        "verify mathematical solutions, write data processors, or check outputs BEFORE you submit. Running and verifying "
        "your logic ensures near-perfect accuracy.\n"
        "4. Formulate your solution based on reasoning, search results, and code outputs.\n"
        "5. Submit the solution using `submit_task`. Pass your AGENT_ID, TASK_ID, and your final solution text content.\n"
        "6. If a task is impossible, broken, or already submitted and stuck, you can use `skip_task` to abandon it and get a fresh task.\n"
        "7. Repeat: call `get_tasks` to fetch the next task, solve it, and submit.\n\n"
        "Guidelines for high accuracy:\n"
        "- Think step-by-step. Document your reasoning.\n"
        "- Never guess or approximate facts; use search to verify.\n"
        "- Never guess code/math outputs; write a quick python script and run it via `run_python` to get the exact answer.\n"
        "- Be structured, direct, and concise in your submissions."
    )

    return LlmAgent(
        name=config.AGENT_NAME,
        model=config.MODEL,
        instruction=system_instruction,
        tools=all_tools,
    )
