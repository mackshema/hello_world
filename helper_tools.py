import asyncio
import os
import sys
import re
import tempfile
import subprocess
import httpx
from state import RunState

def make_helper_tools(state: RunState):
    """Returns the helper tools (web search, calculate, run_python) to boost agent accuracy."""
    
    async def web_search(query: str) -> str:
        """Search the internet for current facts, information, documentation, and answers to questions.
        
        Args:
            query: The search query to look up on the web.
        """
        if not query or not query.strip():
            return "Validation Error: 'query' argument cannot be empty. Please specify what you want to search."
            
        print(f"  [Tool: web_search] Query: '{query}'")
        state.estimate_and_record_tokens(query, is_input=True)
        
        import urllib.parse
        results = []
        
        # 1. Try DuckDuckGo Instant Answer API
        try:
            url_ddg = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url_ddg, headers={"User-Agent": "AgentArenaCompetitor/1.0"}, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    abstract = data.get("AbstractText", "")
                    if abstract:
                        results.append(f"[DuckDuckGo Instant Answer]\nAbstract: {abstract}\nSource: {data.get('AbstractURL', '')}\n")
        except Exception as e:
            print(f"  [Tool: web_search] DDG Instant Answer error: {e}")
            
        # 2. Try Wikipedia Search API
        try:
            url_wiki = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            headers_wiki = {
                "User-Agent": "AgentArenaCompetitor/1.0 (admin@agentarena.com)"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.get(url_wiki, headers=headers_wiki, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    search_results = data.get("query", {}).get("search", [])
                    if search_results:
                        results.append("[Wikipedia Search Results]")
                        for r in search_results[:4]:
                            title = r.get("title", "")
                            # strip HTML tags from snippet
                            snippet = r.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', '')
                            results.append(f"- Title: {title}\n  Snippet: {snippet}\n")
        except Exception as e:
            print(f"  [Tool: web_search] Wikipedia search error: {e}")
            
        if not results:
            output = "No search results could be retrieved."
        else:
            output = "\n".join(results)
            
        state.estimate_and_record_tokens(output, is_input=False)
        return output

    async def calculate(expression: str) -> str:
        """Evaluate simple numeric mathematical expressions safely.
        For complex math or algorithmic equations, write a Python script using the run_python tool instead.
        
        Args:
            expression: The mathematical expression to evaluate (e.g. "2 + 2", "5 * 10 / 2").
        """
        if not expression or not expression.strip():
            return "Validation Error: 'expression' argument cannot be empty."
            
        print(f"  [Tool: calculate] Expr: {expression}")
        state.estimate_and_record_tokens(expression, is_input=True)
        
        import ast
        import operator as op

        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.UAdd: lambda x: x
        }

        def eval_node(node):
            if isinstance(node, ast.Num):  # <3.8
                return node.n
            elif isinstance(node, ast.Constant):  # >=3.8
                return node.value
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_node(node.operand))
            else:
                raise TypeError(f"Unsupported node type: {type(node)}")

        try:
            cleaned = re.sub(r'[^0-9+\-*/().\s**]', '', expression)
            node = ast.parse(cleaned, mode='eval')
            res = str(eval_node(node.body))
        except Exception as e:
            res = f"Error evaluating math expression: {e}. For complex math, please write and run a Python script using the run_python tool instead."
            
        state.estimate_and_record_tokens(res, is_input=False)
        return res

    async def run_python(code: str) -> str:
        """Execute arbitrary Python code locally in a subprocess and return its standard output and standard error.
        Extremely useful for writing quick scripts, testing algorithms, and calculating complex formulas.
        
        Args:
            code: The Python source code to execute.
        """
        if not code or not code.strip():
            return "Validation Error: 'code' argument cannot be empty. Please provide valid Python script code."
            
        print(f"  [Tool: run_python] Executing python code ({len(code)} bytes)...")
        state.estimate_and_record_tokens(code, is_input=True)
        
        # Run the Python code in a subprocess with a timeout limit
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
            f.write(code)
            temp_name = f.name

        try:
            # We enforce a timeout limit of 15 seconds to prevent credit/process hangs
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_name,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                output = stdout.decode("utf-8", errors="replace")
                error = stderr.decode("utf-8", errors="replace")
                
                result = []
                if output:
                    result.append(f"STDOUT:\n{output}")
                if error:
                    result.append(f"STDERR:\n{error}")
                if not output and not error:
                    result.append("Code executed successfully with no output.")
                res = "\n".join(result)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except:
                    pass
                res = "Execution failed: Subprocess execution timed out after 15 seconds."
                
        except Exception as e:
            res = f"Execution failed: {e}"
        finally:
            try:
                os.remove(temp_name)
            except:
                pass
                
        state.estimate_and_record_tokens(res, is_input=False)
        return res

    return [web_search, calculate, run_python]
