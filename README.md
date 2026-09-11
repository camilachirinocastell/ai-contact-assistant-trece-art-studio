# AI Contact Assistant — Trece Art Studio

Stateful AI agent built with LangGraph that classifies incoming client
inquiries and generates an appropriate response, with qualified lead
persistence in SQLite and an interactive web interface built with Streamlit.

**Live URL:** _Coming soon_

## Overview

The agent analyzes each incoming message, extracts key metadata (service
type, level of definition, urgency), and dynamically routes the conversation
toward either a commercial proposal or a polite rejection, based on the
graph's state.

## Tech stack

- Python
- LangChain + LangGraph
- Google Gemini API (`langchain-google-genai`)
- SQLite
- Streamlit
- `python-dotenv` for environment variable management

## Project status

🚧 In progress. See the repository branches for step-by-step feature
development.

## Local installation

```bash
git clone https://github.com/camilachirinocastell/ai-contact-assistant-trece-art-studio.git
cd ai-contact-assistant-trece-art-studio
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY
```

## Environment variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | API key for the Google Gemini model, generated in Google AI Studio |

## Run

```bash
python main.py        # console test
streamlit run app.py  # web interface
```

## Project structure

├── config.py # LLM initialization, decoupled from the rest of the logic
├── prompts.py # Prompt templates
├── database.py # SQLite persistence layer for qualified leads
├── agent.py # AgentState definition and LangGraph state graph
├── main.py # Console entry point for quick testing
└── app.py # Streamlit web interface


## Author

Camila Chirino Castell —
💻 Portfolio: [camilachirinocastell-portfolio.netlify.app](https://camilachirinocastell-portfolio.netlify.app)
🐙 GitHub: [github.com/camilachirinocastell](https://github.com/camilachirinocastell)
👤 LinkedIn: [www.linkedin.com/in/camila-chirino-castell](https://www.linkedin.com/in/camila-chirino-castell)