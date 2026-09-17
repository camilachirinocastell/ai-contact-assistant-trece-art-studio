# config.py
# Decoupled initialization of the LLM client (Gemini).
# If the provider changes in the future, this is the only file to touch.

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Loads the variables defined in .env into the process environment.
# In production the variables are already set at the platform level,
# so load_dotenv() simply finds nothing and doesn't break anything.
load_dotenv()

# Model name centralized here, not repeated in every graph node.
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a ready-to-use LLM client.

    temperature=0.0 by default: for an agent that classifies and
    extracts metadata, we want consistent responses, not creative ones.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not defined. "
            "Check that a .env file exists at the project root "
            "with the key loaded (see .env.example)."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=api_key,
        temperature=temperature,
    )