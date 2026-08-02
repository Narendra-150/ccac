# ─────────────────────────────────────────────────────────────────────
# CLAUDE CERTIFIED ARCHITECT — Domain 1
# Topic: Multi-Agent System with Mock Data (Simulates Real API Calls)
# ────────────────────────────────────────────────────────────────────
#
# COMPLETE WORKING EXAMPLE showing:
#   1. Coordinator spawns specialized subagents via Task tool
#   2. Each subagent has role-scoped tools (mock implementations)
#   3. Real client.messages.create() in the agentic loop
#   4. Results unified back to coordinator
#
# Run: python3 d3_multi_agent_with_mock.py
# ────────────────────────────────────────────────────────────────────

from anthropic import Anthropic
import json
import os
import time
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ────────────────────────────────────────────────────────────────────
# CLIENT INITIALIZATION (matches d2_agent_simple.py pattern)
# ────────────────────────────────────────────────────────────────────
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "test"),
    base_url=os.getenv("ANTHROPIC_BASE_URL", "http://localhost:1234")
)

# ────────────────────────────────────────────────────────────────────
# MOCK DATA (reusable like d2_agent_simple.py's mock_books)
# ────────────────────────────────────────────────────────────────────
MOCK_LIBRARY = {
    "978-0140283297": {"title": "The Timeless Way of Building", "author": "Christopher Alexander", "status": "available", "shelf": "A3", "copies": 4, "year": 1990},
    "978-0061120084": {"title": "The Alchemist", "author": "Paulo Coelho", "status": "checked_out", "shelf": "B1", "copies": 0, "year": 1988},
    "978-0451524935": {"title": "Neuromancer", "author": "William Gibson", "status": "reserved", "shelf": "C2", "copies": 1, "year": 1984},
}

MOCK_REVIEWS = {
    "978-0140283297": {"rating": 4.8, "summary": "Groundbreaking architecture manifesto", "pros": ["Practical", "Thought-provoking"], "cons": ["Dense"]},
    "978-0061120084": {"rating": 4.6, "summary": "Beautiful allegorical tale", "pros": ["Inspirational", "Well-written"], "cons": []},
    "978-0451524935": {"rating": 4.7, "summary": "Revolutionary cyberpunk classic", "pros": ["Influential", "Fast-paced"], "cons": ["Outdated tech refs"]},
}

MOCK_AUTHORS = {
    "Christopher Alexander": {"born": 1936, "nationality": "American", "fields": ["Architecture", "Design"], "notable_works": ["Pattern Language"]},
    "Paulo Coelho": {"born": 1947, "nationality": "Brazilian", "fields": ["Literature"], "notable_works": ["The Alchemist", "Valley of the Wind"]},
    "William Gibson": {"born": 1948, "nationality": "American", "fields": ["Science Fiction"], "notable_works": ["Neuromancer", "Sprawl Trilogy"]},
}

# ────────────────────────────────────────────────────────────────────
# COORDINATOR TOOLS (Task tool to spawn subagents)
# ────────────────────────────────────────────────────────────────────
coordinator_tools = [
    {
        "type": "tool",
        "name": "Task",  # Capital T required!
        "description": "Spawn a specialized subagent to handle a specific task",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Label for subagent"},
                "prompt": {"type": "string", "description": "Task specification"},
                "allowed_tools": {"type": "array", "items": {"type": "string"}},
                "parameters": {"type": "object", "description": "Parameters to pass to subagent"}
            },
            "required": ["description", "prompt", "allowed_tools"]
        }
    }
]

COORDINATOR_SYSTEM = """You are a library research coordinator using Hub-and-Spoke.
Decompose large queries into specialized subagents.
Spawn ALL in ONE message for PARALLEL execution.
Return unified findings."""

# ────────────────────────────────────────────────────────────────────
# MOCK TOOL EXECUTORS (like execute_tool in d2_agent_simple.py)
# ────────────────────────────────────────────────────────────────────
def get_inventory(tool_input: dict) -> dict:
    """Mock inventory lookup with simulated delay (0-5s)."""
    time.sleep(random.uniform(0, 5))  # Simulate API latency
    isbn = tool_input.get("isbn", "")
    if isbn in MOCK_LIBRARY:
        return {"found": True, "data": MOCK_LIBRARY[isbn]}
    return {"found": False, "error": "Book not found"}

def get_reviews(tool_input: dict) -> dict:
    """Mock reviews lookup with simulated delay (0-5s)."""
    time.sleep(random.uniform(0, 5))  # Simulate API latency
    isbn = tool_input.get("isbn", "")
    if isbn in MOCK_REVIEWS:
        return {"found": True, "data": MOCK_REVIEWS[isbn]}
    return {"found": False, "error": "No reviews"}

def get_author_info(tool_input: dict) -> dict:
    """Mock author lookup with simulated delay (0-5s)."""
    time.sleep(random.uniform(0, 5))  # Simulate API latency
    author_name = tool_input.get("author_name", "")
    if author_name in MOCK_AUTHORS:
        return {"found": True, "data": MOCK_AUTHORS[author_name]}
    return {"found": False, "error": "Author not found"}

# ────────────────────────────────────────────────────────────────────
# COORDINATOR AGENT CLASS (replicates d2_agent_simple.py pattern)
# ────────────────────────────────────────────────────────────────────
class LibraryCoordinator:
    """Hub coordinating specialized subagents via Task tool."""
    
    def __init__(self):
        self.subagent_results = []
    
    def create_subagent_config(self, domain: str, isbn: str, tool_name: str) -> dict:
        """Factory for subagent configuration matching ep/e02 notes."""
        return {
            "description": domain,
            "prompt": f"You are a {domain.lower()} specialist. Research {isbn}. Use {tool_name}. Return JSON findings.",
            "allowed_tools": [tool_name],
            "parameters": {"isbn": isbn}
        }
    
    def run(self, isbn: str) -> str:
        """Agentic loop: coordinator spawns subagents, collects results."""
        messages = [{"role": "user", "content": f"Research all info about book: {isbn}"}]
        
        for iteration in range(1, 10):
            # REAL CL API CALL (matches d2_agent_simple.py line 246)
            response = client.messages.create(
                model="claude-opus-4-1-20250807",
                max_tokens=4096,
                system=COORDINATOR_SYSTEM,
                tools=coordinator_tools,
                messages=messages
            )
            
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        return f"{block.text}\n\n## SUBAGENT FINDINGS\n{json.dumps(self.subagent_results, indent=2)}"
                return ""
            
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._handle_task(block)
                        tool_results.append(result)
                messages.append({"role": "user", "content": tool_results})
        
        return "Error: Agent did not complete"
    
    def _handle_task(self, block) -> Dict:
        """Execute Task tool invocation (simulates subagent execution)."""
        tool_name = block.name
        tool_id = block.id
        tool_input = block.input
        
        if tool_name != "Task":
            return {"type": "tool_result", "tool_use_id": tool_id, "is_error": True,
                    "content": json.dumps({"error": f"Unknown tool: {tool_name}"})}
        
        domain = tool_input.get("description", "")
        params = tool_input.get("parameters", {})
        isbn = params.get("isbn", "")
        
        result = {
            "description": domain,
            "status": "completed",
            "findings": self._execute_subagent_tool(domain, isbn)
        }
        self.subagent_results.append(result)
        
        return {"type": "tool_result", "tool_use_id": tool_id, "content": json.dumps(result)}
    
    def _execute_subagent_tool(self, domain: str, isbn: str) -> list:
        """Execute the scoped tool for a subagent domain."""
        if "Inventory" in domain:
            data = get_inventory({"isbn": isbn})
            if data.get("found"):
                return [f"Status: {data['data']['status']}", f"Copies: {data['data']['copies']}", f"Shelf: {data['data']['shelf']}"]
        elif "Review" in domain:
            data = get_reviews({"isbn": isbn})
            if data.get("found"):
                return [f"Rating: {data['data']['rating']}/5", f"Summary: {data['data']['summary']}"]
        elif "Author" in domain:
            if isbn in MOCK_LIBRARY:
                author = MOCK_LIBRARY[isbn].get("author", "")
                data = get_author_info({"author_name": author})
                if data.get("found"):
                    return [f"Author: {author}", f"Nationality: {data['data']['nationality']}"]
        return ["No data found"]


# ────────────────────────────────────────────────────────────────────
# TIMING DEMONSTRATION (shows parallel vs sequential latency)
# ────────────────────────────────────────────────────────────────────
def demonstrate_timing():
    """Demonstrate parallel vs sequential execution timing."""
    print("\n" + "="*70)
    print("PARALLEL vs SEQUENTIAL EXECUTION TIMING")
    print("="*70)
    
    tasks = [
        ("Inventory Specialist", "get_inventory"),
        ("Review Specialist", "get_reviews"),
        ("Author Specialist", "get_author_info")
    ]
    
    # Parallel execution
    print("\n--- PARALLEL (correct pattern) ---")
    print("Sending 3 Task blocks in ONE message...")
    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(lambda t: time.sleep(random.uniform(0,3)) or f"Result: {t}", name): name for name, _ in tasks}
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            print(f"  ✓ {name} complete")
    parallel_time = time.time() - start
    
    # Sequential for comparison
    print("\n--- SEQUENTIAL (anti-pattern) ---")
    print("Sending 1 Task block per turn...")
    start = time.time()
    for name, _ in tasks:
        time.sleep(random.uniform(0, 3))
        print(f"  ✓ {name} complete (waited)")
    sequential_time = time.time() - start
    
    print(f"\n⏱️  Parallel time: {parallel_time:.2f}s")
    print(f"⏱️  Sequential time: {sequential_time:.2f}s")
    print(f"⚡ Speedup: {sequential_time/parallel_time:.1f}x faster with parallel!")


# ────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Show message structure
    print("\n" + "="*70)
    print("HUB-AND-SPOKE: PARALLEL SUBAGENT SPAWNING")
    print("="*70)
    print("""
CORRECT - PARALLEL (Single assistant message with 3 Task blocks):

{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Researching in parallel..."},
    {"type": "tool_use", "name": "Task", "id": "tu_001", "input": {...Inventory config...}},
    {"type": "tool_use", "name": "Task", "id": "tu_002", "input": {...Review config...}},
    {"type": "tool_use", "name": "Task", "id": "tu_003", "input": {...Author config...}}
  ]
}

WRONG - SEQUENTIAL (Multiple turns):

Turn 1: [Task(Inventory)] → wait
Turn 2: [Task(Reviews)] → wait  
Turn 3: [Task(Author)] → wait
""")
    
    # Show example message structure
    isbn = "978-0140283297"
    coordinator = LibraryCoordinator()
    
    print(f"Example: Coordinator spawning 3 subagents for {isbn}")
    print("Each has its own scoped tool (get_inventory, get_reviews, get_author_info)\n")
    
    for tool in ["get_inventory", "get_reviews", "get_author_info"]:
        config = coordinator.create_subagent_config("X Specialist", isbn, tool)
        print(f"  • {config['description']} → allowed_tools: {config['allowed_tools']}")
    
    # Run timing demo
    demonstrate_timing()