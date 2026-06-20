import asyncio
import os
import sys
import re
import config
from state import RunState
from agent import build_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

async def run_turn(runner, session_id, message, state: RunState):
    """Send one message; collect and return the agent's final text reply."""
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
    
    # Estimate and record tokens for the input message
    state.estimate_and_record_tokens(message, is_input=True)
    
    final_reply = ""
    async for event in runner.run_async(session_id=session_id, new_message=content, user_id="default_user"):
        if getattr(event, "turn_complete", False):
            final_reply = event.content.parts[0].text
            break
            
    # Estimate and record tokens for the output response
    if final_reply:
        state.estimate_and_record_tokens(final_reply, is_input=False)
        
    return final_reply

async def run_turn_with_retry(runner, session_id, message, state: RunState, max_retries: int = 5):
    """Sends a message to the runner, automatically retrying on 429 rate limit / quota errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return await run_turn(runner, session_id, message, state)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                # Parse suggested retry delay or default to 60s
                delay = 60.0
                match = re.search(r"Please retry in (\d+\.?\d*)s", err_str)
                if match:
                    delay = float(match.group(1)) + 1.5  # Add 1.5s buffer
                
                print(f"\n[Rate Limit Detected] Gemini API Quota exceeded. (Attempt {attempt}/{max_retries})")
                print(f"Waiting {delay:.1f} seconds for quota to reset before retrying...\n")
                await asyncio.sleep(delay)
            else:
                # Re-raise other client or runtime exceptions
                raise e
    raise RuntimeError(f"Execution failed after {max_retries} attempts due to persistent rate limits.")

async def main():
    # 1. Validate environment configuration on startup
    config.validate_config(exit_on_failure=True)
    
    state = RunState()
    agent = build_agent(state)
    sessions = InMemorySessionService()
    
    print("\n==========================================")
    print(f"Starting Modular Arena Agent: {config.AGENT_NAME}")
    print(f"Stack: {config.AGENT_STACK}")
    print(f"Model: {config.MODEL}")
    print("==========================================\n")

    # Turn 1: Registration
    reg_session_id = f"{state.run_id}_register"
    await sessions.create_session(
        session_id=reg_session_id,
        app_name=config.AGENT_NAME,
        user_id="default_user"
    )
    runner = Runner(agent=agent, session_service=sessions, app_name=config.AGENT_NAME)
    
    print("--- [Turn 1] Registering Agent ---")
    register_message = f"Please register the agent using the register_agent tool with name '{config.AGENT_NAME}' and stack '{config.AGENT_STACK}'."
    reply = await run_turn_with_retry(runner, reg_session_id, register_message, state)
    print(f"[Agent]: {reply}")
    
    if not state.agent_id:
        print("ERROR: Agent registration failed. Cannot proceed without an Agent ID.", file=sys.stderr)
        sys.exit(1)
        
    # Solve tasks sequentially in fresh, clean sessions (Session Reset Strategy)
    task_count = 0
    while task_count < config.MAX_TURNS:
        task_count += 1
        
        # Create a fresh session ID for this specific task
        task_session_id = f"{state.run_id}_task_{task_count}"
        await sessions.create_session(
            session_id=task_session_id,
            app_name=config.AGENT_NAME,
            user_id="default_user"
        )
        
        print(f"\n--- [Task Attempt {task_count}] Starting clean session '{task_session_id}' ---")
        
        task_message = (
            f"Solve the current task. Your Agent ID is '{state.agent_id}'. "
            f"Call get_tasks to retrieve your active task, use accuracy helper tools (web_search, calculate, run_python) "
            f"to solve it accurately, and call submit_task to submit your solution. (Current Level: {state.current_level})"
        )
        
        # Run in clean session with automatic rate limit retry support
        reply = await run_turn_with_retry(runner, task_session_id, task_message, state)
        print(f"[Agent]: {reply}")
        
        # Pause to avoid rate limits
        await asyncio.sleep(2)
        
    print("\n==========================================")
    print("Agent Arena Competitor Loop Completed.")
    print(f"Tasks Attempted: {state.tasks_attempted}")
    print(f"Total Score: {state.total_score}")
    print(f"Final Level: {state.current_level}")
    print(f"Estimated Credits Used: {state.input_tokens} Input, {state.output_tokens} Output")
    print(f"Estimated Total Cost: ${state.estimated_cost:.6f} USD")
    print("==========================================\n")

if __name__ == "__main__":
    asyncio.run(main())
