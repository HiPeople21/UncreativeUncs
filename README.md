# Talentier — LinkedIn Candidate Discovery

An AI-powered recruiter tool that finds real LinkedIn candidates based on technical skills, experience level, and location filters. Built with **React TSX** + **FastAPI** + **LangGraph** + **Claude (Sonnet 4.6)**.

## Architecture

```
React TSX (Vite) → FastAPI → LangGraph Agent → Claude (Sonnet 4.6)
                                    ↓
                              DuckDuckGo Search
                            (real LinkedIn profiles)
```

The LangGraph agent runs a multi-step pipeline:
1. **plan_search** (LLM) — Plans the optimal search query
2. **search_linkedin** (DuckDuckGo) — Fetches real LinkedIn profiles
3. **enrich** (LLM) — Cleans and enriches candidate data
4. **evaluate** (LLM) — Scores and ranks candidates by relevance
5. **decide** — If results are poor, loops back to refine the search

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**

## Setup

### 1. Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variable
cp .env.sample .env
```

Update `.env` with actual API keys.

### 2. Frontend

```bash
cd frontend
npm install
```

## Running

Open **two terminals**:

```bash
# Terminal 1 — Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

## Usage

1. Select **technical skills** (Python, React, AWS, etc.)
2. Choose an **experience level** (Junior → Principal)
3. Pick a **location** (or leave as "Any")
4. Click **Search Candidates**
5. View real LinkedIn profiles, click cards for details, open profiles on LinkedIn

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, Python |
| AI Agent | LangGraph, LangChain |
| LLM | Claude (Sonnet 4.6) |
| Search | DuckDuckGo HTML |
| Styling | Vanilla CSS (dark theme, glassmorphism) |