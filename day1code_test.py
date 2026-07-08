from anthropic import Anthropic

client = Anthropic(
    api_key="lm-studio",
    base_url="http://localhost:1234"
)

response = client.messages.create(
    model="local-model",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.content[0].text)
