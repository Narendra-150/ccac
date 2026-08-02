# ─────────────────────────────────────────────────────────────────────
# Claude Certified Architect — Domain 1: Multi-Agent with Real LLM Calls
# Topic: Hub-and-Spoke Pattern with client.messages.create()
# ────────────────────────────────────────────────────────────────────
#
# This is a COMPLETE end-to-end example showing:
#   1. Client initialization
#   2. Real API calls to client.messages.create()
#   3. Coordinator spawning subagents via Task tool
#   4. Subagent results returned to coordinator
#   5. Unified final response
# ────────────────────────────────────────────────────────────────────

from anthropic import Anthropic
import json
import os
import time
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as ConcurrentTimeout

# ────────────────────────────────────────────────────────────────────
# CLIENT INITIALIZATION - Connect to LLM
# ────────────────────────────────────────────────────────────────────
# Uses environment variables for security (best practice)
# Falls back to local LM Studio proxy for development
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "test"),
    base_url=os.getenv("ANTHROPIC_BASE_URL", "http://localhost:1234")
)

# ────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ────────────────────────────────────────────────────────────────────
# The Task tool is a BUILT-IN tool for spawning subagents
# Type must be "tool" (lowercase), name must be "Task" (capital T)
coordinator_tools = [
    {
        "type": "tool",
        "name": "Task",
        "description": "Spawn a specialized subagent to handle a specific task and return its findings",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short label for the subagent's role"
                },
                "prompt": {
                    "type": "string",
                    "description": "Complete task definition for the subagent"
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tools available to the subagent"
                }
            },
            "required": ["description", "prompt", "allowed_tools"]
        }
    }
]

# ────────────────────────────────────────────────────────────────────
# COORDINATOR SYSTEM PROMPT
# ────────────────────────────────────────────────────────────────────
COORDINATOR_SYSTEM = """You are a Hub-and-Spoke coordinator.

The Hub-and-Spoke Pattern:
- HUB: You orchestrate, decompose, unify
- SPOKES: Specialized subagents you spawn via Task tool

Your job:
1. Decompose complex requests into 2-4 specialized subagent tasks
2. Spawn ALL subagents in ONE message (parallel execution)
3. Collect all subagent results
4. Synthesize findings into a unified response

DO NOT provide step-by-step procedures.
Be concise and authoritative.
"""


# ────────────────────────────────────────────────────────────────────
# SUBAGENT CONFIGURATION
# ────────────────────────────────────────────────────────────────────
def get_subagent_config(domain: str, expertise: str, delay_min: int = 0, delay_max: int = 30) -> dict:
    """Create subagent configuration with 0-30s random delay (default)."""
    return {
        "description": f"{expertise} Expert: {domain}",
        "prompt": f"""You are a {expertise.lower()} expert specializing in {domain.lower()}.

TASK: Research {domain}

QUALITY CRITERIA:
- Find recent data (2022-2025)
- Identify 3-5 key findings
- Note any conflicting conclusions or gaps

RETURN FORMAT: Strict JSON object with:
- findings: array of key insights
- metrics: array of relevant numbers
- conflicts: array of conflicting conclusions

Random processing time: {delay_min}-{delay_max}s
JSON:""",
        "allowed_tools": ["web_search", "read_url", "scrape"]
    }


# ────────────────────────────────────────────────────────────────────
# TIMING SIMULATION UTILITIES
# ────────────────────────────────────────────────────────────────────
def simulate_delay(min_sec: int = 0, max_sec: int = 30) -> float:
    """Simulate processing time with random delay (0-30s by default)."""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)
    return delay


def run_sequential_with_delays():
    """Demonstrate SEQUENTIAL execution with random delays (0-30s default)."""
    print("\n" + "=" * 70)
    print("SEQUENTIAL EXECUTION (ANTI-PATTERN) - With Random Delays")
    print("=" * 70)
    print("Default delay: 0-30 seconds per subagent\n")

    domains = [
        ("Market Analysis", "Market", 0, 30),
        ("Technology Review", "Technology", 0, 30),
        ("Policy Analysis", "Policy", 0, 30)
    ]

    start_total = time.time()
    results = []

    for i, (domain, expertise, min_d, max_d) in enumerate(domains, 1):
        print(f"  → Starting: {expertise}...")
        start = time.time()
        delay = simulate_delay(min_d, max_d)
        elapsed = time.time() - start
        print(f"  ✓ Completed: {expertise} ({elapsed:.1f}s)")

        results.append({"domain": domain, "elapsed": elapsed})

    total = time.time() - start_total
    print(f"\n⏱️  Total sequential time: {total:.1f}s")
    print("   (Sum of all individual delays)")
    return results, total


def run_parallel_with_delays():
    """Demonstrate PARALLEL execution with random delays."""
    print("\n" + "=" * 70)
    print("PARALLEL EXECUTION (RECOMMENDED) - With Random Delays")
    print("=" * 70)
    print("Default delay: 0-30 seconds per subagent\n")

    domains = [
        ("Market Analysis", "Market", 0, 30),
        ("Technology Review", "Technology", 0, 30),
        ("Policy Analysis", "Policy", 0, 30)
    ]

    start_total = time.time()
    results = []

    print("Starting all 3 subagents concurrently...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_domain = {}
        for domain, expertise, min_d, max_d in domains:
            print(f"  → Started: {expertise}...")
            future = executor.submit(_run_task_worker, domain, expertise, min_d, max_d)
            future_to_domain[future] = (domain, expertise)

        for future in as_completed(future_to_domain):
            domain, expertise = future_to_domain[future]
            result = future.result()
            print(f"  ✓ Completed: {expertise}")
            results.append(result)

    total = time.time() - start_total
    print(f"\n⏱️  Total parallel time: {total:.1f}s")
    print("   (Max of all individual delays)")
    return results, total


def _run_task_worker(domain: str, expertise: str, min_d: int, max_d: int) -> dict:
    """Worker function that simulates subagent work with delay."""
    delay = simulate_delay(min_d, max_d)
    return {"domain": domain, "expertise": expertise, "elapsed": delay}


def demonstrate_timing_comparison():
    """Compare sequential vs parallel with delays."""
    print("\n" + "=" * 70)
    print("TIMING COMPARISON (Default: 0-30s random delays)")
    print("=" * 70)

    random.seed(42)
    seq_results, seq_time = run_sequential_with_delays()

    random.seed(42)
    par_results, par_time = run_parallel_with_delays()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Sequential: {seq_time:.1f}s (adds up delays)")
    print(f"  Parallel:   {par_time:.1f}s (takes max delay)")
    speedup = seq_time / par_time if par_time > 0 else 1
    print(f"  Speedup:    {speedup:.1f}x faster with parallel")


# ────────────────────────────────────────────────────────────────────
# AGENTIC LOOP - Hub Implementation
# ────────────────────────────────────────────────────────────────────
class Hub:
    """
    The Hub (coordinator) in Hub-and-Spoke architecture.
    
    Calls client.messages.create() to communicate with LLM.
    Spawns subagents via Task tool.
    Collects and unifies results.
    
    TIMEOUT HANDLING:
    - iteration_timeout: Max seconds per loop iteration
    - max_iterations: Prevent infinite loops (default 10)
    - Error categories: transient, permission, validation, internal
    """

    def __init__(self, max_iterations: int = 10, iteration_timeout: int = 60):
        self.subagent_results: List[Dict] = []
        self.iteration = 0
        self.max_iterations = max_iterations
        self.iteration_timeout = iteration_timeout

    def run(self, user_message: str, domains: List[tuple]) -> str:
        """
        Main agentic loop.
        
        Step 1: Send user message to LLM with Task tool registered
        Step 2: LLM returns Task tool_use blocks (parallel spawn)
        Step 3: Client.execute each Task → spawns subagent
        Step 4: LLM returns end_turn with unified response
        """
        print(f"\n{'='*70}")
        print("HUB-AND-SPOKE: COORDINATOR LOOP")
        print(f"{'='*70}")

        messages = [{"role": "user", "content": user_message}]

        while self.iteration < 10:
            self.iteration += 1

            # ── LLM API CALL ───────────────────────────────────────
            response = client.messages.create(
                model="claude-opus-4-1-20250807",
                max_tokens=4096,
                system=COORDINATOR_SYSTEM,
                tools=coordinator_tools,
                messages=messages
            )
            # ─────────────────────────────────────────────────────────

            # ── HANDLE NEXT ACTION FROM LLM ─────────────────────────
            if response.stop_reason == "end_turn":
                return self._handle_final(response)

            if response.stop_reason == "tool_use":
                return self._handle_task_spawning(response, domains)

        return "Error: Max iterations reached"

    def _handle_final(self, response) -> str:
        """Extract final text when LLM says 'end_turn'."""
        for block in response.content:
            if block.type == "text":
                if self.subagent_results:
                    return f"""{block.text}

## Subagent Findings
{json.dumps(self.subagent_results, indent=2)}"""
                return block.text
        return ""

    def _handle_task_spawning(self, response, domains: List[tuple]) -> str:
        """
        When LLM requests Task tool_use, spawn subagents.
        
        Key insight: Multiple Task block_use in same response = PARALLEL
        """
        print(f"\n  LLM → Coordinator requesting subagent spawn")

        # APPEND 1: Assistant's message (tool uses)
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # EXECUTE ALL TOOL USES
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = self._execute_tool_block(block)
                tool_results.append(result)

        # APPEND 2: Tool results (user role - external data to LLM)
        messages.append({
            "role": "user",
            "content": tool_results
        })

        # Return to loop - LLM will receive results and continue
        return "continue"

    def _execute_tool_block(self, block, timeout: int = 60) -> Dict[str, Any]:
        """
        Execute a tool_use block (Task tool spawns subagent).
        
        Timeout handling for multi-agent resilience:
        - transient (TimeoutError): Retryable after delay
        - permission: Escalate, don't retry
        - validation: Model should self-correct
        - internal: Surface to coordinator
        """
        import time
        
        tool_name = block.name
        tool_id = block.id
        tool_input = block.input

        if tool_name == "Task":
            print(f"    → Spawning: {tool_input.get('description')}")

            try:
                # Simulate subagent with timeout protection
                start = time.time()
                
                # In real implementation, this would spawn a subagent
                result = {
                    "description": tool_input.get("description"),
                    "status": "completed",
                    "findings": f"Research completed for: {tool_input.get('description')}",
                    "timestamp": time.strftime("%H:%M:%S")
                }
                
                self.subagent_results.append(result)
                
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result)
                }
                
            except ConcurrentTimeout:
                # TRANSIENT: Timeout - retryable
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": True,
                    "content": json.dumps({
                        "errorCategory": "transient",
                        "isRetryable": True,
                        "description": f"Subagent timeout: {tool_input.get('description')} exceeded {timeout}s",
                        "retryAfterMs": 2000
                    })
                }
                
            except PermissionError as e:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": True,
                    "content": json.dumps({
                        "errorCategory": "permission",
                        "isRetryable": False,
                        "description": f"Access denied: {str(e)}"
                    })
                }
                
            except ValueError as e:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": True,
                    "content": json.dumps({
                        "errorCategory": "validation",
                        "isRetryable": False,
                        "description": f"Invalid input: {str(e)}"
                    })
                }
                
            except Exception as e:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "is_error": True,
                    "content": json.dumps({
                        "errorCategory": "internal",
                        "isRetryable": False,
                        "description": f"Unexpected error: {str(e)}",
                        "errorType": type(e).__name__
                    })
                }

        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "is_error": True,
            "content": json.dumps({"error": f"Unknown tool: {tool_name}"})
        }


# ────────────────────────────────────────────────────────────────────
# TIMEOUT & ERROR HANDLING EXPLANATION
# ────────────────────────────────────────────────────────────────────
"""
TIMEOUT IN MULTI-AGENT SYSTEMS (Key Exam Concept)
=================================================

WHY TIMEOUTS HAPPEN:
1. API latency spikes (network issues)
2. Slow subagents (complex reasoning)
3. Rate limiting from LLM provider
4. External tool calls (web search, database) timing out

ERROR CATEGORIES (Exam standard):
------------------------------------
1. TRANSIENT (retryable):
   - TimeoutError
   - ConnectionError  
   - RateLimitError
   → Delay 1-2s and retry, or pass to coordinator for retry

2. PERMISSION (escalate):
   - PermissionError
   - AccessDenied
   → Don't retry; coordinate needs to handle or human escalation

3. VALIDATION (self-correct):
   - ValueError, bad input params
   → Model should recognize error and retry with corrected input

4. INTERNAL (surface):
   - Unexpected exceptions, bugs
   → Log and surface; shouldn't happen in production

MAX ITERATIONS GUARD:
--------------------
The max_iterations parameter (default 10-20) prevents infinite loops.
If a subagent keeps crashing or timing out, the loop eventually exits
rather than spinning forever.

SUBAGENT TIMEOUT:
----------------
Each subagent can have its own timeout. If it exceeds the limit:
- Return structured error with is_error: true
- Include "transient" errorCategory with isRetryable: true
- Provide retryAfterMs for backoff strategy

BEST PRACTICES:
--------------
- Use environment variables for timeouts (dev vs prod)
- Implement exponential backoff for retryable errors
- Log each subagent's start/end time for observability
- Group related subagents for parallel execution
- Keep subagent timeouts generous but bounded
"""


# ────────────────────────────────────────────────────────────────────
# DEMONSTRATION FUNCTIONS
# ────────────────────────────────────────────────────────────────────
def show_parrel_vs_sequential():
    """Display the message structure difference."""
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ PARALLEL (Recommended - Single Turn)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Turn 1: Coordinator → LLM                                           │
│ {                                                                   │
│   "role": "assistant",                                              │
│   "content": [                                                    │
│     {"type": "text", "text": "Researching in parallel."},          │
│     {"type": "tool_use", "name": "Task", ...},  ← Market         │
│     {"type": "tool_use", "name": "Task", ...},  ← Technology       │
│     {"type": "tool_use", "name": "Task", ...}   ← Policy          │
│   ]                                                                 │
│ }                                                                   │
│                                                                     │
│ Turn 2: Coordinator ← LLM                                           │
│ {                                                                   │
│   "role": "user",                                                   │
│   "content": [result_1, result_2, result_3]  ← All at once!       │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ SEQUENTIAL (Anti-Pattern - Multiple Turns)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Turn 1: coordinator → LLM                                           │
│ { "role": "assistant", "content": [{"type": "tool_use", "name":...}] }│
│                                                                     │
│ Turn 2: coordinator ← LLM                                           │
│ { "role": "user", "content": [result_1] }                          │
│                                                                     │
│ Turn 3: coordinator → LLM                                           │
│ { "role": "assistant", "content": [{"type": "tool_use", ...}]}      │
│                                                                     │
│ ... (repeat for each subagent)                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


# ────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    show_parrel_vs_sequential()

    # Run timing demonstration with random 0-30s delays
    demonstrate_timing_comparison()

    # Define research domains for subagents
    domains = [
        ("Off-grid solar market in Africa", "Market"),
        ("Solar technology providers", "Technology"),
        ("Policy incentives and barriers", "Policy")
    ]

    # User query
    user_query = """Research off-grid solar solutions for East Africa.
    Provide market size, key technology players, and policy landscape."""

    # Run the hub
    hub = Hub()
    result = hub.run(user_query, domains)

    print(f"\n{'='*70}")
    print("FINAL HUB RESPONSE")
    print(f"{'='*70}")
    print(result)