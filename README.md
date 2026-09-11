# LangGraph Research and Writing Pipeline

An interactive Python workflow that combines web research, human review, and
LLM-assisted writing using LangGraph, Groq, and Tavily.

## Key features

- Researches a user request with Tavily web search.
- Pauses for human approval, editing, or rejection.
- Retries rejected research up to two times with reviewer feedback.
- Produces a concise final paragraph from the reviewed research.
- Keeps workflow state in an in-memory LangGraph checkpointer.

## Prerequisites

- Python 3.11 or newer
- A Groq API key
- A Tavily API key

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install langgraph langchain langchain-core langchain-groq \
  langchain-tavily python-dotenv
cp .env.example .env
```

Set `GROQ_API_KEY` and `TAVILY_API_KEY` in `.env`. Never commit `.env`; rotate
any credentials that may already have been exposed.

## Usage

```bash
python main.py
```

At the review prompt, choose:

- `a` to approve the research.
- `e` to replace it with edited notes.
- `r` to provide feedback and request another research pass.

## Directory structure

```text
.
├── .env.example
├── .gitignore
├── main.py
└── README.md
```

## Security note

API credentials belong in environment variables, not source code or committed
configuration. The included `.gitignore` excludes local secrets and virtual
environments.

## Suggested initial commit

`feat: initial repository setup with LangGraph research workflow`
