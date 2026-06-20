import json
import re
import httpx
import config
from state import RunState
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

async def mcp_call(tool: str, args: dict, state: RunState) -> str:
    """Open a fresh MCP session, call one tool, return text result."""
    try:
        transport = StreamableHttpTransport(url=config.MCP_ENDPOINT)
        async with Client(transport, name="arena-agent") as c:
            result = await c.call_tool(tool, args)
        
        text_content = "\n".join(
            getattr(b, "text", "")
            for b in result.content
            if getattr(b, "text", None)
        )
        
        # Track credit usage by estimating tokens
        state.estimate_and_record_tokens(json.dumps(args), is_input=True)
        state.estimate_and_record_tokens(text_content, is_input=False)
        
        return text_content
    except Exception as e:
        err_msg = f"MCP Network Error executing tool '{tool}': {e}"
        print(f"  [MCP Error] {err_msg}")
        return err_msg


def make_arena_tools(state: RunState):
    """Returns the four Arena tool functions with state captured via closure."""
    
    async def register_agent(name: str, stack: str) -> str:
        """Register this agent in the Agent Arena. Call once at start. Returns AGENT_ID.
        
        Args:
            name: The agent's name on the leaderboard.
            stack: Description of the agent's tech stack.
        """
        if not name or not name.strip():
            return "Validation Error: 'name' argument cannot be empty. Please provide a valid agent name."
        if not stack or not stack.strip():
            return "Validation Error: 'stack' argument cannot be empty. Please provide a valid stack description."
            
        print(f"  [Tool: register_agent] Name: {name}, Stack: {stack}")
        result = await mcp_call("register_agent",
            {"idToken": config.ARENA_ID_TOKEN, "name": name, "stack": stack}, state)
        
        m = re.search(r"AGENT_ID:\s*(\S+)", result)
        if m: 
            state.agent_id = m.group(1)
            print(f"  [Tool: register_agent] Successfully registered Agent ID: {state.agent_id}")
        return result

    async def get_tasks(agent_id: str) -> str:
        """Fetch the current assigned task from the Arena. Returns JSON with task id, title, description, level, points.
        
        Args:
            agent_id: The registered agent ID.
        """
        if not agent_id or not agent_id.strip():
            return "Validation Error: 'agent_id' argument is empty. You must register first using register_agent."
            
        print(f"  [Tool: get_tasks] Agent ID: {agent_id}")
        result = await mcp_call("get_tasks",
            {"idToken": config.ARENA_ID_TOKEN, "agentId": agent_id}, state)
            
        try:
            data = json.loads(result)
            if "id" in data: 
                state.task_id = data["id"]
                print(f"  [Tool: get_tasks] Received Task ID: {state.task_id} | Title: {data.get('title')}")
            else:
                print(f"  [Tool: get_tasks] Warning: No active task assigned. Response: {result}")
        except Exception as e:
            print(f"  [Tool: get_tasks] Error parsing JSON response: {e}")
        return result

    async def submit_task(agent_id: str, task_id: str, content: str) -> str:
        """Submit your answer/solution for AI evaluation. Scored 0-100. Score >= 70 triggers LEVEL_UP.
        Each task can only be submitted once.
        
        Args:
            agent_id: The registered agent ID.
            task_id: The current task ID.
            content: The text content containing your final answer or response.
        """
        if not agent_id or not agent_id.strip():
            return "Validation Error: 'agent_id' cannot be empty. Please verify you registered first."
        if not task_id or not task_id.strip():
            return "Validation Error: 'task_id' cannot be empty. Please call get_tasks to retrieve a task first."
        if not content or not content.strip():
            return "Validation Error: 'content' to submit cannot be empty. Please specify your solution."
            
        print(f"  [Tool: submit_task] Submitting solution for Task {task_id}...")
        result = await mcp_call("submit_task", {
            "idToken": config.ARENA_ID_TOKEN, "agentId": agent_id,
            "taskId": task_id, "content": content,
            "metadata": {"agent_name": config.AGENT_NAME, "model": config.MODEL},
        }, state)
        
        print(f"  [Tool: submit_task] Response: {result}")
        try:
            score_m = re.search(r"score[:\s]*(\d+)", result, re.IGNORECASE)
            score = int(score_m.group(1)) if score_m else 0
            levelled_up = "LEVEL_UP" in result or "level up" in result.lower() or "passed" in result.lower()
            state.record(state.current_level, f"Task {task_id}", score, levelled_up)
            if levelled_up or score >= 70:
                state.task_id = ""
        except Exception as e:
            print(f"  [Tool: submit_task] Error recording statistics: {e}")
            
        return result

    async def skip_task(agent_id: str, task_id: str) -> str:
        """Abandon/skip the current task without penalty. Unlocks a fresh task.
        
        Args:
            agent_id: The registered agent ID.
            task_id: The current task ID.
        """
        if not agent_id or not agent_id.strip():
            return "Validation Error: 'agent_id' is required to skip."
        if not task_id or not task_id.strip():
            return "Validation Error: 'task_id' is required to skip."
            
        print(f"  [Tool: skip_task] Skipping Task {task_id}")
        result = await mcp_call("skip_task",
            {"idToken": config.ARENA_ID_TOKEN, "agentId": agent_id, "taskId": task_id}, state)
        state.task_id = ""
        return result

    return [register_agent, get_tasks, submit_task, skip_task]
