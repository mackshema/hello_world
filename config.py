import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Mandatory environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ARENA_ID_TOKEN = os.getenv("ARENA_ID_TOKEN")

# Optional configurations (with defaults)
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "https://agent-arena-623774504237.asia-southeast1.run.app/mcp")
AGENT_NAME = os.getenv("AGENT_NAME", "ApexAgent_v1")
# Sanitize AGENT_NAME to ensure it is a valid Python identifier
import re
AGENT_NAME = re.sub(r'[^a-zA-Z0-9_]', '_', AGENT_NAME)
if AGENT_NAME and not AGENT_NAME[0].isalpha() and AGENT_NAME[0] != '_':
    AGENT_NAME = '_' + AGENT_NAME

AGENT_STACK = os.getenv("AGENT_STACK", "Python / ADK / Gemini")
MODEL = os.getenv("MODEL", "gemini-2.0-flash")

try:
    MAX_TURNS = int(os.getenv("MAX_TURNS", "20"))
except ValueError:
    MAX_TURNS = 20

def validate_config(exit_on_failure: bool = True) -> bool:
    """Validate project configuration variables.
    
    Prints clean warnings and instructions if configurations are missing or invalid.
    """
    errors = []
    
    if not GEMINI_API_KEY:
        errors.append(
            "GEMINI_API_KEY is not set.\n"
            "  -> Please obtain a free API key from https://aistudio.google.com and add it to your .env file."
        )
        
    if not ARENA_ID_TOKEN or ARENA_ID_TOKEN == "<paste your Firebase token here>":
        errors.append(
            "ARENA_ID_TOKEN is missing or is the default placeholder.\n"
            "  -> To get a token: Sign in to the Arena web app, open DevTools -> Application -> Storage -> Local Storage,\n"
            "     copy the value of ID_TOKEN, and paste it into your .env file."
        )
        
    if errors:
        print("\n================ CONFIGURATION ERROR ================", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        print("=====================================================\n", file=sys.stderr)
        if exit_on_failure:
            sys.exit(1)
        return False
        
    return True
