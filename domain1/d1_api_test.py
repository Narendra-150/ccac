from anthropic import Anthropic

# Initialize Anthropic client configured for LM Studio local proxy
# api_key and base_url route requests to a local LLM server
client = Anthropic(
    api_key="lm-studio",
    base_url="http://localhost:1234"
)

# Send a chat completion request to the local model
response = client.messages.create(
    model="minicpm5-1b-claude-opus-fable5-thinking",        # Name of the local model served by LM Studio
    max_tokens=1024,            # Maximum number of tokens to generate
    system="You are a helpful assistant.",  # System prompt setting assistant behavior
    messages=[
        {"role": "user", "content": "Hello, how are you?"}  # User message payload
    ]
)

# Print the text content of the first response message
print(response.content[0].text)

