# ─────────────────────────────────────────────────────────────────────
# Claude Certified Architect — Domain 1: Building with Claude API
# Topic: Hub-and-Spoke Multi-Agent Architecture with Timing Demo
# ─────────────────────────────────────────────────────────────────────
#
# ARCHITECTURE PATTERN: Hub-and-Spoke
# ────────────────────────────────────────────────────────────────────
# HUB (Coordinator): Orchestrates, decomposes, unifies results
# SPOKES (Subagents): Specialized roles with scoped tools and prompts
#
# TIMING DEMO: Shows why PARALLEL > SEQUENTIAL
#   - Each subagent has 0-30s random delay (default for testing)
#   - Parallel: total time ≈ max(individual times)
#   - Sequential: total time = sum of all delays
# ────────────────────────────────────────────────────────────────────

from anthropic import Anthropic
import json
import os
import time
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ────────────────────────────────────────────────────────────────────
# CLIENT INITIALIZATION
# ────────────────────────────────────────────────────────────────────
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY", "test"),
    base_url=os.getenv("ANTHROPIC_BASE_URL", "http://localhost:1234")
)

# ────────────────────────────────────────────────────────────────────
# TOOLS
# ────────────────────────────────────────────────────────────────────
coordinator_tools = [
    {
        "type": "tool",
        "name": "Task",  # Capital T is required!
        "description": "Spawn a specialized subagent to execute a specific task",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Label for subagent"},
                "prompt": {"type": "string", "description": "Task specification"},
                "allowed_tools": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["description", "prompt", "allowed_tools"]
        }
    }
]

COORDINATOR_SYSTEM = """You are a research coordinator implementing Hub-and-Spoke.
Decompose tasks into 2-4 specialized subagents.
Spawn ALL in ONE message for parallel execution.
Return unified findings."""


def simulate_subagent_delay(min_sec: int = 0, max_sec: int = 60) -> float:
    """Simulate processing time for a subagent (0-60 seconds)."""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)
    return delay


def get_subagent_config(domain: str, expertise: str) -> dict:
    """Create scoped subagent configuration."""
    return {
        "description": f"{expertise} expert: {domain}",
        "prompt": f"""You are a {expertise.lower()} expert.
Research {domain}.
Return JSON: findings, metrics, conflicts.""",
        "allowed_tools": ["web_search", "read_url"]
    }


# ────────────────────────────────────────────────────────────────────
# TIMING DEMO FUNCTIONS
# ────────────────────────────────────────────────────────────────────
def run_sequential_timing():
    """
    SEQUENTIAL pattern: Each subagent waits for previous to complete.
    Total time = sum of all delays.
    """
    print("\n" + "=" * 70)
    print("SEQUENTIAL EXECUTION (ANTI-PATTERN)")
    print("=" * 70)
    print("Each subagent completes before the next starts...\n")

    domains = [
        ("Market Analysis: East Africa solar adoption", "Market"),
        ("Technology Review: Off-grid solutions", "Technology"),
        ("Policy Environment: Regulatory landscape", "Policy")
    ]

    total_delay = 0
    results = []

    for i, (domain, expertise) in enumerate(domains, 1):
        print(f"  [{i}/3] {expertise} expert starting...")
        start = time.time()
        delay = simulate_subagent_delay(0, 30)  # 0-30s random delay
        elapsed = time.time() - start
        total_delay += delay

        result = {
            "domain": domain,
            "expertise": expertise,
            "completed_at": f"+{elapsed:.1f}s"
        }
        results.append(result)
        print(f"        ✓ Completed in {elapsed:.1f}s")

    print(f"\n⏱️  Total sequential time: {total_delay:.1f}s (sum of all)")
    return results, total_delay


def run_parallel_timing():
    """
    PARALLEL pattern: All subagents run concurrently.
    Total time = max of individual delays.
    """
    print("\n" + "=" * 70)
    print("PARALLEL EXECUTION (RECOMMENDED)")
    print("=" * 70)
    print("All subagents run at the same time...\n")

    domains = [
        ("Market Analysis: East Africa solar adoption", "Market"),
        ("Technology Review: Off-grid solutions", "Technology"),
        ("Policy Environment: Regulatory landscape", "Policy")
    ]

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for i, (domain, expertise) in enumerate(domains):
            future = executor.submit(
                _run_task_with_timing, domain, expertise, i + 1
            )
            futures[future] = (domain, expertise)

        for future in as_completed(futures):
            domain, expertise = futures[future]
            result, _ = future.result()
            results.append(result)
            print(f"  ✓ {expertise} complete")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total parallel time: {elapsed:.1f}s (max of all)")
    return results, elapsed


def _run_task_with_timing(domain: str, expertise: str, order: int):
    """Worker function with timing."""
    delay = simulate_subagent_delay(0, 30)
    result = {
        "domain": domain,
        "expertise": expertise,
        "completed_at": f"+{delay:.1f}s"
    }
    return result, delay


# ────────────────────────────────────────────────────────────────────
# COORDINATOR IMPLEMENTATION
# ────────────────────────────────────────────────────────────────────
class CoordinatorAgent:
    """Hub that spawns subagents via Task tool."""

    def __init__(self):
        self.subagent_results: List[Dict] = []

    def run(self, user_request: str) -> str:
        """Execute the coordination loop with REAL client.messages.create()."""
        messages = [{"role": "user", "content": user_request}]

        for iteration in range(1, 10):
            # ── REAL LLM API CALL ─────────────────────────────────────
            response = client.messages.create(
                model="claude-opus-4-1-20250807",
                max_tokens=4096,
                system=COORDINATOR_SYSTEM,
                tools=coordinator_tools,
                messages=messages
            )
            # ───────────────────────────────────────────────────────────

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        if self.subagent_results:
                            return f"{block.text}\n\nSubagent Summary:\n{json.dumps(self.subagent_results, indent=2)}"
                        return block.text
                return ""

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_task(block)
                        tool_results.append(result)

                messages.append({"role": "user", "content": tool_results})

        return "Error: Max iterations reached"

    def _execute_task(self, block) -> Dict:
        """Execute a Task tool_use block."""
        if block.name == "Task":
            print(f"  ✓ Spawned subagent: {block.input.get('description')}")

            result = {
                "description": block.input.get("description"),
                "status": "completed",
                "findings": f"Research for: {block.input.get('description')}",
                "timestamp": "2024-01-15T10:30:00Z"
            }
            self.subagent_results.append(result)

            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)
            }

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "is_error": True,
            "content": json.dumps({"error": f"Unknown tool: {block.name}"})
        }


def show_timing_comparison():
    """Run both patterns and compare timings."""
    print("\n" + "=" * 70)
    print("TIMING COMPARISON (with 0-30s default random delays per subagent)")
    print("=" * 70)

    random.seed(42)

    print("\n" + "-" * 70)
    print("Running SEQUENTIAL demo...")
    print("-" * 70)
    seq_results, seq_time = run_sequential_timing()

    print("\n" + "-" * 70)
    print("Running PARALLEL demo...")
    print("-" * 70)
    par_results, par_time = run_parallel_timing()

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"  Sequential time: {seq_time:.1f}s")
    print(f"  Parallel time:   {par_time:.1f}s")
    speedup = seq_time / par_time if par_time > 0 else 1
    print(f"  Speedup:         {speedup:.1f}x faster with parallel")


def show_message_structures():
    """Display JSON message structures for both patterns."""
    print("\n" + "=" * 70)
    print("MESSAGE STRUCTURE REFERENCE")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ PARALLEL (CORRECT) - Single assistant message with multiple Tasks    │
├─────────────────────────────────────────────────────────────────────┤
│ {                                                                    │
│   "role": "assistant",                                               │
│   "content": [                                                       │
│     {"type": "text", "text": "Researching 3 domains in parallel."}, │
│     {"type": "tool_use", "name": "Task", "id": "tu_001",             │
│      "input": {"description": "Market", ...}},                      │
│     {"type": "tool_use", "name": "Task", "id": "tu_002",             │
│      "input": {"description": "Technology", ...}},                 │
│     {"type": "tool_use", "name": "Task", "id": "tu_003",             │
│      "input": {"description": "Policy", ...}}                       │
│   ]                                                                │
│ }                                                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ SEQUENTIAL (ANTI-PATTERN) - Separate turns for each Task           │
├─────────────────────────────────────────────────────────────────────┤
│ Turn 1: { "role": "assistant",                                    │
│          "content": [{"type": "tool_use", "name": "Task", ...}] }   │
│ Turn 2: { "role": "user", "content": [result_1] }                  │
│ Turn 3: { "role": "assistant", ... Task(Technology) }              │
│ ... (repeat for each)                                              │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    show_message_structures()
    show_timing_comparison()

    print("\n" + "=" * 70)
    print("KEY EXAM TAKEAWAYS")
    print("=" * 70)
    print("""
1. PARALLEL TIMING: Total time = max(subagent times)
   - Delays: 0-30s random per subagent (default)
   - Wall-clock = slowest one only
   - Speedup is dramatic for longer delays

2. SEQUENTIAL TIMING: Total time = sum(subagent times)
   - Wall-clock = all delays added up
   - Each subagent blocks the next

3. PARALLEL PATTERN REQUIRES:
   - Multiple Task blocks in ONE assistant message
   - Single content array with all tool_use blocks
   - NOT separate turns for each Task

4. Task tool name: "Task" (capital T)
   - Must be exactly this in coordinator_tools
   - Subagents should NOT have Task in allowed_tools
""")