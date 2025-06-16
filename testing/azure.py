import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# Load environment variables from .env file
load_dotenv(override=True)


async def test_azure():
    # Debug: Print environment variables to verify
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    print(f"DEBUG: API Key: {api_key}")
    print(f"DEBUG: Endpoint: {endpoint}")
    print(f"DEBUG: Deployment: {deployment}")

    if not all([api_key, endpoint, deployment]):
        raise ValueError(
            "Missing required environment variables: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, or AZURE_OPENAI_DEPLOYMENT"
        )

    client = AsyncAzureOpenAI(
        api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-15-preview"
    )
    response = await client.chat.completions.create(
        model=deployment, messages=[{"role": "user", "content": "hi"}]
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(test_azure())
