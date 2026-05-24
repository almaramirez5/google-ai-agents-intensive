import os
import asyncio
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY not found. Please set it in your environment or .env file.")

async def main():
    retry_config = types.HttpRetryOptions(attempts=3, exp_base=5, initial_delay=1, http_status_codes=[429, 500, 503, 504])
    base_model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

    print("Initializing AI Travel Agency Specialists...")

    # THE INTAKE AGENT 
    # Extracts the core info and saves it to 'trip_context'
    intake_agent = Agent(
        name="Receptionist",
        model=base_model,
        instruction="Extract the destination and duration from the user's prompt. Output ONLY a short summary like: 'Destination: X. Duration: Y days. Budget: Z EUR.'",
        output_key="trip_context" 
    )

    # SPECIALISTS
    culinary_agent = Agent(
        name="CulinaryExpert",
        model=base_model,
        instruction="Based on this trip ({trip_context}), suggest 3 iconic local food experiences. Provide a total estimated cost in EUR.",
        output_key="culinary_plan" 
    )

    lodging_agent = Agent(
        name="LodgingExpert",
        model=base_model,
        instruction="Based on this trip ({trip_context}), suggest 3 accommodation options. Provide estimated cost per night in EUR.",
        output_key="lodging_plan"
    )

    activities_agent = Agent(
        name="ActivitiesExpert",
        model=base_model,
        instruction="Based on this trip ({trip_context}), suggest 3 must-do activities. Provide total estimated ticket costs in EUR.",
        output_key="activities_plan"
    )

    research_phase = ParallelAgent(
        name="ResearchPhase",
        sub_agents=[culinary_agent, lodging_agent, activities_agent]
    )

    # THE SYNTHESIZER 
    planner_agent = Agent(
        name="ItineraryPlanner",
        model=base_model,
        instruction="""You are a master travel planner. Create a cohesive day-by-day itinerary for {trip_context} based ONLY on these expert proposals:
        - Culinary: {culinary_plan}
        - Lodging: {lodging_plan}
        - Activities: {activities_plan}
        
        Ensure the final output is formatted beautifully as a cohesive travel guide."""
    )

    # THE MASTER PIPELINE 
    travel_pipeline = SequentialAgent(
        name="TravelAgencyPipeline",
        sub_agents=[intake_agent, research_phase, planner_agent]
    )

    # --- EXECUTION ---
    user_request = "I want to go interrailing through Montenegro, Albania, Greece and Croatia. I will be there for 14 days, with my friend."
    print(f"\n--- NEW USER REQUEST ---\n{user_request}\n")
    print("Processing...")

    runner = InMemoryRunner(agent=travel_pipeline)
    final_itinerary = await runner.run_debug(user_request)
    
    print(f"\n[FINAL ITINERARY]\n{final_itinerary}")

if __name__ == "__main__":
    asyncio.run(main())
