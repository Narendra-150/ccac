from anthropic import Anthropic
import json

# ---------------------------------------------------------------------------
# Claude Certified Architect — Domain 1: Building with Claude API
# Topic: Tool Use + Agentic Loop
# ---------------------------------------------------------------------------
#
# This script demonstrates how to build an agent using Claude's Messages API.
# The agent uses the tool_use feature to look up library book information,
# demonstrating the full agentic loop pattern.
#
# Key Anthropic API concepts covered:
#   1. client.messages.create() — primary API entry point
#   2. tools — registering callable tools Claude can invoke
#   3. stop_reason — determining whether model finished ("end_turn")
#                     or wants to call tools ("tool_use")
#   4. tool_result blocks — feeding execution results back to Claude
#   5. Error taxonomy for tool results: transient/permission/validation/internal
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLIENT INITIALIZATION
# ---------------------------------------------------------------------------
# - api_key: required by the SDK; "test" is used here for local proxy setups.
# - base_url: override to point to a local LLM server (e.g., LM Studio).
#   This is common in exam scenarios where you test without a real API key.
# ---------------------------------------------------------------------------
client = Anthropic(
    api_key="test",
    base_url="http://localhost:1234"
)


# ---------------------------------------------------------------------------
# TOOL DEFINITION
# ---------------------------------------------------------------------------
# The `tools` list is sent to the model on every request. Claude examines
# the user's message and decides whether any tool is relevant.
#
# When Claude decides to use a tool, it returns a stop_reason of "tool_use"
# with content blocks containing:
#   - block.name   → which tool to call (must match the name below)
#   - block.id     → unique ID that MUST be echoed back in the tool_result
#   - block.input  → arguments dict matching the input_schema
#
# Tool definition schema:
#   - name:        tool identifier (must match your Python function name)
#   - description: tells Claude WHEN and HOW to use this tool
#   - input_schema: JSON Schema describing required/optional parameters
#   - type: always "object" for top-level; nested types use dicts/lists.
# ---------------------------------------------------------------------------
tools = [
    {
        "name": "lookup_book",
        "description": (
            "Look up a library book by ISBN. "
            "Returns availability status, shelf location, and total copies. "
            "Use this when the customer asks if a book is available or where to find it. "
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "book_id": {
                    "type": "string",
                    "description": "The ISBN-13 of the book. (e.g. '978-0061120084')"
                }
            },
            "required": ["book_id"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL EXECUTION
# ---------------------------------------------------------------------------
# execute_tool runs the actual Python logic behind a tool call.
# It MUST return a JSON string — not a dict. The API wraps the tool_result
# structure and expects `content` to be a string.
#
# In production, this would query a real database or external service.
# Here we use a mock catalog for demonstration.
# ---------------------------------------------------------------------------
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return its result as a JSON string."""
    if tool_name == "lookup_book":
        book_id = tool_input.get("book_id", "")

        # Mock library catalog data for demonstration
        mock_books = {
            "978-0140283297": {"status": "available",  "shelf": "A3", "copies": 4},
            "978-0061120084": {"status": "checked_out", "shelf": "B1", "copies": 0},
            "978-0451524935": {"status": "reserved",    "shelf": "C2", "copies": 1},
        }

        if book_id in mock_books:
            return json.dumps(mock_books[book_id])
        else:
            return json.dumps({"error": f"Book {book_id} not found in catalog"})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ---------------------------------------------------------------------------
# TOOL CALL HANDLER
# ---------------------------------------------------------------------------
# handle_tool_call wraps execute_tool and handles failures gracefully.
# It returns a complete "tool_result" dict with:
#   - type: "tool_result"   (required by the API)
#   - tool_use_id:          the ID from Claude's tool_use block (required)
#   - is_error:             optional flag; True signals Claude this call failed
#   - content:              execution result as a JSON string
#
# Error categories guide Claude on how to respond:
#   transient  → temporary (network/DB timeout). Claude can retry.
#   permission → access denied. Retrying won't help; escalate to human.
#   validation → bad input parameters. Claude should self-correct and retry.
#   internal   → unexpected error. Surface to coordinator/human for review.
# ---------------------------------------------------------------------------
def handle_tool_call(tool_name: str, tool_id: str, tool_input: dict) -> dict:
    """
    Execute one tool call and return a structured tool_result dict.
    Structured error categories let Claude decide: retry, self-correct, or escalate.
      transient  → temporary infrastructure hiccup; safe to retry after delay
      permission → access denied; retrying won't help; escalate
      validation → bad input parameters; model should self-correct before retrying
      internal   → unexpected error; surface to coordinator / human operator
    """
    print(f"  -> Calling tool: {tool_name}({tool_input})")

    try:
        content = execute_tool(tool_name, tool_input)
        print(f"  <- Result: {content}")
        return {
            "type":        "tool_result",
            "tool_use_id": tool_id,
            "content":     content,
        }

    except TimeoutError as e:
        # transient — infrastructure hiccup; retryable after a short delay
        print(f"  <- ERROR: TimeoutError on {tool_name}")
        return {
            "type":        "tool_result",
            "tool_use_id": tool_id,
            "is_error":    True,
            "content":     json.dumps({
                "errorCategory": "transient",
                "isRetryable":   True,
                "description":   f"Timeout calling {tool_name}: {str(e)}",
                "retryAfterMs":  2000,
            }),
        }

    except PermissionError as e:
        # permission — agent lacks access; retrying won't help; escalate
        print(f"  <- ERROR: PermissionError on {tool_name}")
        return {
            "type":        "tool_result",
            "tool_use_id": tool_id,
            "is_error":    True,
            "content":     json.dumps({
                "errorCategory": "permission",
                "isRetryable":   False,
                "description":   f"Access denied for {tool_name}: {str(e)}",
            }),
        }

    except ValueError as e:
        # validation — bad input parameters; model should self-correct, not retry
        print(f"  <- ERROR: ValueError on {tool_name}")
        return {
            "type":        "tool_result",
            "tool_use_id": tool_id,
            "is_error":    True,
            "content":     json.dumps({
                "errorCategory": "validation",
                "isRetryable":   False,
                "description":   f"Invalid input for {tool_name}: {str(e)}",
            }),
        }

    except Exception as e:
        # internal — unexpected; log and surface to coordinator/human
        print(f"  <- ERROR: {type(e).__name__} on {tool_name}")
        return {
            "type":        "tool_result",
            "tool_use_id": tool_id,
            "is_error":    True,
            "content":     json.dumps({
                "errorCategory": "internal",
                "isRetryable":   False,
                "description":   f"Unexpected error in {tool_name}: {str(e)}",
            }),
        }


# ---------------------------------------------------------------------------
# AGENTIC LOOP
# ---------------------------------------------------------------------------
# This is the core pattern for building agents with the Anthropic Messages API.
#
# Step-by-step flow:
#   1. Initialize messages[] with the user's request
#   2. Call client.messages.create() with tools=...
#   3. Inspect response.stop_reason:
#        - "end_turn"   => model produced final text; extract and return it
#        - "tool_use"   => model wants to call one or more tools
#   4. For each tool_use block:
#        a. Execute the tool locally (via execute_tool + handle_tool_call)
#        b. Build a tool_result block (echoing the tool_use_id)
#        c. Append tool_result blocks to messages[] with role="user"
#   5. Loop back to step 2 until "end_turn" or max iterations reached
#
# WHY messages[] history matters:
#   Claude is stateless — it only knows what you send in the messages[]
#   array on each request. Every turn (assistant messages + tool results)
#   must be appended so Claude has full context for the next API call.
#
# MAX_ITERATIONS guard:
#   Prevents infinite loops if Claude gets stuck repeatedly calling tools.
#   50 is a safe upper bound for most simple agents.
# ---------------------------------------------------------------------------
def run_agent(user_message: str) -> str:
    """
    Run the agentic loop until Claude produces a final text answer.
    Returns the final assistant text response.
    """
    # Key concept: messages[] is the conversation state.
    # It starts with the user's request and grows with every API round-trip.
    messages = [
        {"role": "user", "content": user_message}
    ]

    MAX_ITERATIONS = 50  # Safety guard against infinite tool-use loops
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # Invoke Claude's Messages API.
        # Passing tools=tools makes tool_use eligible for this turn.
        # Passing messages=messages gives Claude the full conversation context.
        response = client.messages.create(
            model="minicpm5-1b-claude-opus-fable5-thinking",
            max_tokens=4096,
            tools=tools,              # Makes tools eligible for this turn
            messages=messages         # Full conversation history to date
        )

        # ---------------------------------------------------------------
        # CASE 1: Model produced a final text response
        # ---------------------------------------------------------------
        # When stop_reason is "end_turn", the model is done.
        # response.content contains text blocks (and maybe tool_use blocks
        # from earlier turns, but not in this final response).
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""  # end_turn with no text block (rare but possible)

        # ---------------------------------------------------------------
        # CASE 2: Model requested one or more tool invocations
        # ---------------------------------------------------------------
        # stop_reason is "tool_use" — Claude wants to call tools before
        # producing a final answer. We must:
        #   a. Append the assistant's tool_use request to messages[]
        #   b. Execute each tool
        #   c. Append the results back as a user message
        #   d. Loop again so Claude can produce the final answer
        if response.stop_reason == "tool_use":
            # STEP 1: Append the assistant message (containing tool_use blocks)
            # Why? Claude is stateless. On the next API call, it must "see"
            # its own tool_use requests to understand what results it received.
            messages.append({
                "role": "assistant",
                "content": response.content   # Contains tool_use blocks
            })

            # STEP 2: Execute each tool_use block the model requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # block.name  -> which tool to run (e.g. "lookup_book")
                    # block.id    -> unique ID Claude will reference later
                    # block.input -> dict of parameters (e.g. {"book_id": "..."})
                    tool_results.append(handle_tool_call(block.name, block.id, block.input))

            # STEP 3: Feed execution results back to Claude as a user message
            # The tool_result blocks must reference tool_use_id so Claude knows
            # which result corresponds to which tool call.
            # Role is "user" because tool results are treated as user-provided
            # context from Claude's perspective.
            messages.append({
                "role": "user",
                "content": tool_results
            })

    return "Error: agent did not complete within the iteration limit"


# ---------------------------------------------------------------------------
# ENTRY POINT — Multiple example queries
# ---------------------------------------------------------------------------
# Each run demonstrates the same agentic loop pattern with different inputs.
# All queries go through LM Studio at localhost:1234.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: Valid ISBN — book is checked out (0 copies)
    print("=== Example 1: Query checked-out book ===")
    answer = run_agent("Do you have the book with ISBN 978-0061120084?")
    print(f"\nFinal answer: {answer}")

    # Example 2: Valid ISBN — book is available (4 copies)
    print("\n=== Example 2: Query available book ===")
    answer = run_agent("Do you have the book with ISBN 978-0140283297?")
    print(f"\nFinal answer: {answer}")

    # Example 3: Valid ISBN — book is reserved (1 copy)
    print("\n=== Example 3: Query reserved book ===")
    answer = run_agent("Do you have the book with ISBN 978-0451524935?")
    print(f"\nFinal answer: {answer}")

    # Example 4: Invalid ISBN — book not found in catalog
    # Demonstrates the error path: tool returns error dict, Claude sees it
    # in the tool_result and produces a graceful final answer.
    print("\n=== Example 4: Query unknown ISBN ===")
    answer = run_agent("Do you have the book with ISBN 978-0140283217?")
    print(f"\nFinal answer: {answer}")
