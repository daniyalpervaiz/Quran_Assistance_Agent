import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, AsyncOpenAI
import os
import asyncio

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI-API-KEY")

# Setup Streamlit page
st.set_page_config(page_title="Quran Topic Assistant", layout="centered")
st.title("📖 Quran's Topic AI Assistant")
st.markdown("Ask any question, and get answers based on what the Quran says (with Roman Urdu translation of the Ayat).")

# Initialize OpenAI-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define agent
agents = Agent(
    name="Quran's Topic Agents",
    instructions=(
        "You are Quran's Topic Agent. Your task is to answer questions strictly based "
        "on what the Quran says about the topic the user asks.give Arabic ayat Always include a translation "
        "of the relevant Quranic Ayat in Roman Urdu. Do not provide irrelevant or unrelated information."
    )
)

# Async function for running the agent
async def run_agent_async(question):
    return await Runner.run(
        agents,
        input=question,
        run_config=run_config
    )

# UI input for user question
user_question = st.text_input("❓ What would you like to ask from the Quran?")

# Handle response
if st.button("Get Answer") and user_question:
    with st.spinner("Thinking..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(run_agent_async(user_question))
            st.success("Here's what the Quran says:")
            st.write(response.final_output)
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
