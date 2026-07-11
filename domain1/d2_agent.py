from anthropic import Anthropic
import json

# ---------------------------------------------------------------------------
# Claude Certified Architect — Domain 1: Building with Claude API
# Topic: Tool Use + Agentic Loop
# ---------------------------------------------------------------------------
#
# This script demonstrates how to build an agent using Claude's Messages API.
# The agent uses the tool_use feature to act as a car insurance advisor,
# helping customers find plans and compare coverage options.
#
# Key Anthropic API concepts covered:
#   1. client.messages.create() — primary API entry point
#   2. tools — registering callable tools Claude can invoke
#   3. stop_reason — determining whether model finished ("end_turn")
#                     or wants to call tools ("tool_use")
#   4. tool_result blocks — feeding execution results back to Claude
#   5. Multiple tools in one agentic turn
#   6. Error taxonomy: transient/permission/validation/internal
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
# TOOL DEFINITIONS
# ---------------------------------------------------------------------------
# The `tools` list is sent to the model on every request. Claude examines
# the user's message and decides which tools (if any) to invoke.
#
# When Claude decides to use tools, it returns stop_reason="tool_use"
# with content blocks containing:
#   - block.name   → which tool to call (must match definition below)
#   - block.id     → unique ID that MUST be echoed back in the tool_result
#   - block.input  → arguments dict matching the input_schema
#
# Tool definition schema:
#   - name:        tool identifier (must match your Python function name)
#   - description: tells Claude WHEN and HOW to use this tool
#   - input_schema: JSON Schema describing required/optional parameters
#   - type: always "object" for top-level; nested types use dicts/lists.
#
# Exam tip: Claude can call MULTIPLE tools in a single turn. Each tool_use
# block has its own id. You must execute all of them and return all results
# before looping back to the API. Order is preserved.
# ---------------------------------------------------------------------------
tools = [
    {
        "name": "get_insurance_quotes",
        "description": (
            "Get available car insurance plans. "
            "Use this when the customer asks for available plans, wants to see options, "
            "or asks 'what plans do you have'. "
            "Returns a list of plans with monthly premium, deductible, and coverage features."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "brand": {
                    "type": "string",
                    "description": "Car manufacturer / brand. Leave empty for all brands."
                },
                "min_premium": {
                    "type": "number",
                    "description": "Minimum monthly premium filter. Optional."
                },
                "max_premium": {
                    "type": "number",
                    "description": "Maximum monthly premium filter. Optional."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_faq",
        "description": (
            "Look up frequently asked questions about car insurance. "
            "Use this when the customer asks about claim process, cancellation policy, "
            "or how to add a driver. "
            "Returns a relevant answer string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic keyword: 'claim', 'cancel', 'add_driver', 'renew'"
                }
            },
            "required": ["topic"]
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
# Here we use mock data for demonstration.
# ---------------------------------------------------------------------------
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return its result as a JSON string."""
    if tool_name == "get_insurance_quotes":
        brand_filter = tool_input.get("brand", "").lower()
        min_premium = tool_input.get("min_premium", 0)
        max_premium = tool_input.get("max_premium", float("inf"))

        # Mock insurance plan catalog — rich structured data for exam demo
        all_plans = [
            {
                "plan_id": "PLAN-A1",
                "insurer": "SafeDrive Co.",
                "plan_name": "Basic Shield",
                "brand": "Toyota",
                "coverage_type": "Third Party Liability",
                "monthly_premium": 45.00,
                "annual_premium": 540.00,
                "deductible": 1000.00,
                "features": ["accident", "third_party"],
                "exclusions": ["theft", "natural_disaster", "roadside_assistance"],
                "eligible_driver_age": "25+",
                "no_claim_bonus": "0%",
                "free_addon_drivers": 0
            },
            {
                "plan_id": "PLAN-A2",
                "insurer": "SafeDrive Co.",
                "plan_name": "Premium Guard",
                "brand": "Toyota",
                "coverage_type": "Comprehensive",
                "monthly_premium": 95.00,
                "annual_premium": 1140.00,
                "deductible": 500.00,
                "features": ["accident", "theft", "third_party", "roadside_assistance", "natural_disaster"],
                "exclusions": ["rental_coverage"],
                "eligible_driver_age": "21+",
                "no_claim_bonus": "20%",
                "free_addon_drivers": 1
            },
            {
                "plan_id": "PLAN-B1",
                "insurer": "RoadShield Ins.",
                "plan_name": "Value Plan",
                "brand": "Honda",
                "coverage_type": "Third Party Liability",
                "monthly_premium": 38.00,
                "annual_premium": 456.00,
                "deductible": 1500.00,
                "features": ["accident", "third_party"],
                "exclusions": ["theft", "natural_disaster", "roadside_assistance", "rental_coverage"],
                "eligible_driver_age": "25+",
                "no_claim_bonus": "0%",
                "free_addon_drivers": 0
            },
            {
                "plan_id": "PLAN-B2",
                "insurer": "RoadShield Ins.",
                "plan_name": "Elite Coverage",
                "brand": "Honda",
                "coverage_type": "Comprehensive + Zero DEP",
                "monthly_premium": 120.00,
                "annual_premium": 1440.00,
                "deductible": 250.00,
                "features": ["accident", "theft", "third_party", "roadside_assistance", "natural_disaster"],
                "exclusions": ["rental_coverage", "personal_ belongings"],
                "eligible_driver_age": "18+",
                "no_claim_bonus": "25%",
                "free_addon_drivers": 2
            },
            {
                "plan_id": "PLAN-C1",
                "insurer": "QuickCover",
                "plan_name": "Economy Plan",
                "brand": "Maruti Suzuki",
                "coverage_type": "Third Party Liability",
                "monthly_premium": 32.00,
                "annual_premium": 384.00,
                "deductible": 2000.00,
                "features": ["accident", "third_party"],
                "exclusions": ["theft", "natural_disaster", "roadside_assistance", "rental_coverage"],
                "eligible_driver_age": "25+",
                "no_claim_bonus": "0%",
                "free_addon_drivers": 0
            },
            {
                "plan_id": "PLAN-C2",
                "insurer": "QuickCover",
                "plan_name": "Smart Plan",
                "brand": "Maruti Suzuki",
                "coverage_type": "Comprehensive",
                "monthly_premium": 68.00,
                "annual_premium": 816.00,
                "deductible": 750.00,
                "features": ["accident", "theft", "third_party", "roadside_assistance"],
                "exclusions": ["natural_disaster", "rental_coverage"],
                "eligible_driver_age": "21+",
                "no_claim_bonus": "15%",
                "free_addon_drivers": 1
            }
        ]

        # Apply brand filter
        results = [
            plan for plan in all_plans
            if (not brand_filter or plan["brand"].lower() == brand_filter)
            and min_premium <= plan["monthly_premium"] <= max_premium
        ]

        if results:
            return json.dumps({"plans": results})
        else:
            return json.dumps({"plans": [], "message": "No plans match the given criteria."})

    elif tool_name == "get_faq":
        topic = tool_input.get("topic", "").lower().strip()

        faqs = {
            "claim": {
                "topic": "Filing a Claim",
                "answer": (
                    "1. Immediately inform the insurer via app/phone. "
                    "2. File FIR if theft or major accident. "
                    "3. Submit required documents: RC, DL, police report (if any), photos of damage. "
                    "4. A surveyor will be assigned within 48 hours. "
                    "5. Claim is settled after surveyor assessment. "
                    "Typical turnaround: 7–15 working days."
                )
            },
            "cancel": {
                "topic": "Cancellation Policy",
                "answer": (
                    "1. Cancellation is free within 15 days of purchase (free-look period). "
                    "2. After 15 days, a short-rate cancellation fee applies. "
                    "3. Pro-rata refund is calculated on the unused premium minus charges. "
                    "4. Cancellation is not allowed after 80% of the policy period has elapsed."
                )
            },
            "add_driver": {
                "topic": "Adding a Driver",
                "answer": (
                    "1. Log in to your policy dashboard. "
                    "2. Go to 'Add Driver' under Policy Details. "
                    "3. Enter the new driver's license number and date of birth. "
                    "4. Premium may be updated based on the driver's age and history. "
                    "5. Coverage for the new driver starts within 24 hours of approval."
                )
            },
            "renew": {
                "topic": "Renewal Process",
                "answer": (
                    "1. Renewal reminder is sent via SMS/email 30 days before expiry. "
                    "2. You can renew online using the policy number. "
                    "3. NCB (No Claim Bonus) is carried forward if renewed within 90 days of expiry. "
                    "4. No inspection is required for continuous renewals without a claim."
                )
            }
        }

        # Find the best matching topic (allow partial match)
        if topic in faqs:
            return json.dumps(faqs[topic])
        else:
            for key in faqs:
                if key in topic or topic in key:
                    return json.dumps(faqs[key])
            return json.dumps({
                "topic": topic,
                "answer": "I couldn't find a FAQ for that topic. Please try 'claim', 'cancel', 'add_driver', or 'renew'."
            })

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
# Structured error categories guide Claude's decision-making:
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
#   4. For EACH tool_use block in the response:
#        a. Execute the tool locally (via execute_tool + handle_tool_call)
#        b. Collect a tool_result dict (echoing the tool_use_id)
#   5. Append ALL tool_result dicts to messages[] with role="user"
#   6. Loop back to step 2 until "end_turn" or max iterations reached
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
    # messages[] is the conversation state. It starts with the user's
    # request and grows with every API round-trip.
    messages = [
        {"role": "user", "content": user_message}
    ]

    MAX_ITERATIONS = 50  # Safety guard against infinite tool-use loops
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # Invoke Claude's Messages API.
        # - tools=tools makes all registered tools eligible for this turn.
        # - messages=messages gives Claude the full conversation history.
        response = client.messages.create(
            model="minicpm5-1b-claude-opus-fable5-thinking",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # ---------------------------------------------------------------
        # CASE 1: Model produced a final text response
        # ---------------------------------------------------------------
        # When stop_reason is "end_turn", the model is done reasoning.
        # response.content contains text blocks. Extract and return it.
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
        #   b. Execute ALL tools the model requested
        #   c. Collect ALL tool_result dicts
        #   d. Append ALL results back as a single user message
        #   e. Loop again so Claude can produce the final answer
        #
        # Exam tip: Claude may request MULTIPLE tools in one turn
        # (parallel tool calls). Iterate through every content block and
        # execute each tool_use block before sending results back.
        # -------------------------------------------------------------------
        if response.stop_reason == "tool_use":
            # STEP 1: Append assistant's tool_use request to history
            # Why? Claude is stateless. On the next API call, it must "see"
            # its own tool_use requests to correlate with the results.
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # STEP 2: Execute ALL tool_use blocks in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # block.name  -> which tool to run
                    # block.id    -> unique ID Claude will reference later
                    # block.input -> dict of parameters
                    print(f"  -> Tool call: {block.name} with input {block.input}")
                    tool_results.append(handle_tool_call(block.name, block.id, block.input))

            # STEP 3: Feed ALL execution results back as a single user message
            # Each tool_result block must reference its matching tool_use_id
            # so Claude knows which result corresponds to which request.
            # Role is "user" because tool results are treated as external data
            # from Claude's perspective.
            messages.append({
                "role": "user",
                "content": tool_results
            })

    return "Error: agent did not complete within the iteration limit"


# ---------------------------------------------------------------------------
# ENTRY POINT — Multiple example queries
# ---------------------------------------------------------------------------
# Each run demonstrates a different scenario in the car insurance domain.
# All queries go through the local proxy at localhost:1234.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: Ask for Toyota insurance plans
    # Expected: Claude calls get_insurance_quotes with brand="Toyota"
    # Returns 2 plans with full details (premium, deductible, features, etc.)
    print("=== Example 1: Query Toyota insurance plans ===")
    answer = run_agent("I have a Toyota Camry. Show me all available insurance plans.")
    print(f"\nFinal answer: {answer}\n")

    # Example 2: Filter by premium range
    # Expected: Claude calls get_insurance_quotes with premium filters
    # Demonstrates parameter passing and filtering logic
    print("=== Example 2: Filter by budget ===")
    answer = run_agent("Find me plans under $80/month for a Honda Civic.")
    print(f"\nFinal answer: {answer}\n")

    # Example 3: Ask about claim process
    # Expected: Claude calls get_faq with topic="claim"
    # Returns structured answer with step-by-step claim process
    print("=== Example 3: FAQ — How do I file a claim? ===")
    answer = run_agent("How do I file a claim if my car is damaged in an accident?")
    print(f"\nFinal answer: {answer}\n")

    # Example 4: Ask about cancellation
    # Expected: Claude calls get_faq with topic="cancel"
    # Returns cancellation policy with refund details
    print("=== Example 4: FAQ — Cancellation policy ===")
    answer = run_agent("Can I cancel my policy? What are the rules?")
    print(f"\nFinal answer: {answer}\n")

    # Example 5: Complex multi-turn query (quotes + FAQ in one session)
    # Expected: Claude may call get_insurance_quotes first, then based on
    # the results, ask for more info or call get_faq
    print("=== Example 5: Comparison request ===")
    answer = run_agent("Compare SafeDrive's Premium Guard with RoadShield's Elite Coverage for a Honda.")
    print(f"\nFinal answer: {answer}\n")

    # Example 6: Ask about adding drivers
    print("=== Example 6: FAQ — Adding a new driver ===")
    answer = run_agent("How can I add my spouse as a driver to my existing policy?")
    print(f"\nFinal answer: {answer}\n")
