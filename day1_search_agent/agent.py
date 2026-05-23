import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY not found. Please set it in your environment or .env file.")

async def main():
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )

    # 1. Define agent
    root_agent = Agent(
        name="helpful_assistant",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        description="A simple agent that can answer general questions.",
        instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
        tools=[google_search],
    )
  
    print("✅ Root Agent defined.")

    runner = InMemoryRunner(agent=root_agent)

    query = "What is the current weather in Malaga?"
    print(f"\nUser > {query}\n---")
    
    response = await runner.run_debug(query)
    print(f"\nAgent Response > {response}")

if __name__ == "__main__":
    asyncio.run(main())
