import litellm
import os

response = litellm.completion(
    model="openai/Qwen/Qwen2.5-72B-Instruct",  # add `openai/` prefix to model so litellm knows to route to OpenAI
    api_key=os.getenv("OPENAI_API_KEY"),  # api key to your openai compatible endpoint
    api_base=os.getenv(
        "OPENAI_API_BASE"
    ),  # set API Base of your Custom OpenAI Endpoint
    messages=[
        {
            "role": "user",
            "content": "Hey, how's it going?",
        }
    ],
)
print(response)
