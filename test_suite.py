import asyncio
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add workspace directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from state import RunState
from helper_tools import make_helper_tools
from arena_tools import make_arena_tools


class TestArenaAgentModularSystem(unittest.TestCase):
    
    def test_state_token_and_cost_estimation(self):
        """Test that RunState correctly estimates tokens and updates estimated costs."""
        state = RunState()
        self.assertEqual(state.input_tokens, 0)
        self.assertEqual(state.output_tokens, 0)
        self.assertEqual(state.estimated_cost, 0.0)
        
        # Test input estimation: "hello world" (11 chars -> 2 tokens)
        state.estimate_and_record_tokens("hello world", is_input=True)
        self.assertEqual(state.input_tokens, 2)
        self.assertEqual(state.output_tokens, 0)
        
        # Test output estimation: "hello output world" (18 chars -> 4 tokens)
        state.estimate_and_record_tokens("hello output world", is_input=False)
        self.assertEqual(state.input_tokens, 2)
        self.assertEqual(state.output_tokens, 4)
        
        # Manual token recording and pricing verification
        state.record_tokens(1000000, 1000000)
        # Cost for 1M input is $0.075, 1M output is $0.30
        # Plus the tiny fraction from initial 2 and 4 tokens
        expected_cost = 0.075 + 0.30 + ((2 / 1000000) * 0.075) + ((4 / 1000000) * 0.30)
        self.assertAlmostEqual(state.estimated_cost, expected_cost, places=6)

    def test_config_sanitization(self):
        """Test config AGENT_NAME sanitization to valid Python identifier."""
        # Sanitize normal names
        self.assertTrue(config.AGENT_NAME.isidentifier())
        
        # Test sanitization pattern
        import re
        bad_name = "Apex-Agent v1.0!"
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', bad_name)
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized
        self.assertEqual(sanitized, "Apex_Agent_v1_0_")
        self.assertTrue(sanitized.isidentifier())

    def test_config_validation(self):
        """Test config.validate_config behaves correctly when keys are missing or present."""
        with patch("config.GEMINI_API_KEY", ""), patch("config.ARENA_ID_TOKEN", ""):
            res = config.validate_config(exit_on_failure=False)
            self.assertFalse(res)
            
        with patch("config.GEMINI_API_KEY", "some_key"), patch("config.ARENA_ID_TOKEN", "some_token"):
            res = config.validate_config(exit_on_failure=False)
            self.assertTrue(res)

    def test_calculate_tool(self):
        """Test mathematical expressions evaluator (calculate)."""
        state = RunState()
        helper_tools = make_helper_tools(state)
        web_search_tool, calculate_tool, run_python_tool = helper_tools
        
        # Basic calculations
        loop = asyncio.get_event_loop()
        self.assertEqual(loop.run_until_complete(calculate_tool("2 + 3 * 4")), "14")
        self.assertEqual(loop.run_until_complete(calculate_tool("(10 - 2) / 2")), "4.0")
        self.assertEqual(loop.run_until_complete(calculate_tool("2 ** 3")), "8")
        self.assertEqual(loop.run_until_complete(calculate_tool("-5 + 3")), "-2")
        
        # Empty expression validation
        err_msg = loop.run_until_complete(calculate_tool(""))
        self.assertIn("Validation Error", err_msg)
        
        # Safe execution check (unsupported nodes should fail or be filtered out)
        unsafe_msg = loop.run_until_complete(calculate_tool("__import__('os').system('dir')"))
        self.assertIn("Error evaluating math expression", unsafe_msg)

    def test_run_python_tool(self):
        """Test local python script runner tool with subprocess."""
        state = RunState()
        helper_tools = make_helper_tools(state)
        web_search_tool, calculate_tool, run_python_tool = helper_tools
        loop = asyncio.get_event_loop()
        
        # Basic standard output execution
        code = "print('Hello Test Sandbox')"
        res = loop.run_until_complete(run_python_tool(code))
        self.assertIn("STDOUT:", res)
        self.assertIn("Hello Test Sandbox", res)
        
        # Standard error execution
        code_err = "import sys; print('This is an error', file=sys.stderr)"
        res_err = loop.run_until_complete(run_python_tool(code_err))
        self.assertIn("STDERR:", res_err)
        self.assertIn("This is an error", res_err)
        
        # Timeout handling: should abort after 15s limit
        code_timeout = "import time; time.sleep(20)"
        # Patch/mock wait_for to raise TimeoutError immediately to speed up test execution
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            res_to = loop.run_until_complete(run_python_tool(code_timeout))
            self.assertIn("timed out after 15 seconds", res_to)

    def test_web_search_tool_structure(self):
        """Test web search tool execution with fallback APIs."""
        state = RunState()
        helper_tools = make_helper_tools(state)
        web_search_tool, calculate_tool, run_python_tool = helper_tools
        loop = asyncio.get_event_loop()
        
        # Empty search query validation
        err_msg = loop.run_until_complete(web_search_tool(""))
        self.assertIn("Validation Error", err_msg)
        
        # Perform a live search to verify Wikipedia/DDG fallback integrations
        res = loop.run_until_complete(web_search_tool("Python Programming Language"))
        # Should return text containing results
        self.assertTrue(len(res) > 20)

    def test_arena_tools_validation(self):
        """Test Arena tools parameter validation gates."""
        state = RunState()
        arena_tools = make_arena_tools(state)
        register_agent, get_tasks, submit_task, skip_task = arena_tools
        loop = asyncio.get_event_loop()
        
        # Validation for empty name
        res_reg = loop.run_until_complete(register_agent("", "Python"))
        self.assertIn("Validation Error", res_reg)
        
        # Validation for empty stack
        res_reg2 = loop.run_until_complete(register_agent("ApexAgent", ""))
        self.assertIn("Validation Error", res_reg2)
        
        # Validation for get_tasks with empty agent ID
        res_tasks = loop.run_until_complete(get_tasks(""))
        self.assertIn("Validation Error", res_tasks)
        
        # Validation for submit_task with empty values
        res_sub = loop.run_until_complete(submit_task("", "t-1", "sol"))
        self.assertIn("Validation Error", res_sub)
        
        res_sub2 = loop.run_until_complete(submit_task("a-1", "", "sol"))
        self.assertIn("Validation Error", res_sub2)
        
        res_sub3 = loop.run_until_complete(submit_task("a-1", "t-1", ""))
        self.assertIn("Validation Error", res_sub3)


if __name__ == "__main__":
    unittest.main()
