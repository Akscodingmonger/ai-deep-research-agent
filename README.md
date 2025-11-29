# ai-deep-research-agent

A LangGraph-based deep research agent that plans queries, searches the web via Tavily, and generates structured, source-grounded research reports. Includes a lightweight evaluation suite with both heuristic checks and LLM-as-a-judge scoring.

---

## 🚀 Quickstart (Fastest Way to Run)

```bash
git clone https://github.com/<your-username>/ai-deep-research-agent.git
cd ai-deep-research-agent
pip install -r requirements.txt
python deep_research.py # for the agent
python evals.py  # for the evals
```

Create your `.env` file:

```ini
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

---

## Overview

This project implements a deep research agent that:

- Accepts a natural-language query  
- Plans and executes web searches using Tavily  
- Uses LangGraph to coordinate tool usage and reasoning steps  
- Generates a structured, multi-section research report grounded in retrieved sources  
- Includes evaluation tools (heuristics + LLM-as-a-judge)  

Runs fully in the command line — no frontend required.

---

## Features

### LangGraph Workflow
- Research planning  
- Web search retrieval  
- Report generation  

### Structured Output  
1. Executive Summary  
2. Key Findings  
3. Analysis / Discussion  
4. Limitations & Open Questions  
5. Sources (with URLs)

### Additional Capabilities
- Automatic citation indexing (e.g., [1], [2], [3])  
- Structure & grounding checks  
- LLM-as-a-Judge scoring (1–5)  
- Support for multiple queries per session  

---

## Requirements

- Python 3.12+  
- `OPENAI_API_KEY`  
- `TAVILY_API_KEY`  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ai-deep-research-agent.git
cd ai-deep-research-agent
```

### 2. Add a `.env` file

```ini
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 3. (Optional) Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.\.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the research agent

```bash
python deep_research.py
```

Example queries:

- What are the biggest safety challenges facing autonomous vehicles?  
- How is GenAI transforming early-stage startups?  
- What are the environmental risks of large-scale data centers?  
- How do LLMs impact cybersecurity in enterprise systems?  

Outputs a full structured research report.

---

### Run the evaluation tool

```bash
python evals.py
```

Evaluations include:

- Generated research report  
- Structure & grounding checks  
- LLM-as-a-judge scoring (1–5 with justification)  
- Ability to evaluate multiple queries  

---

## Project Structure

```
ai-deep-research-agent/
│
├── deep_research.py        # LangGraph agent and report generator
├── evals.py                # Evaluation suite (heuristics + LLM judge)
├── requirements.txt        # Dependencies
├── .env                    # API keys (ignored by Git)
├── .gitignore              # Keeps venv and secrets out of GitHub
└── README.md               # Project documentation
```

---

## Notes

- `.env` is ignored via `.gitignore`  
- No secrets or venv folders are tracked  
- Code is simple, transparent, and easy to run  

---

## License

MIT License.
