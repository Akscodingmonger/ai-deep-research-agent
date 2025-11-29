# ai-deep-research-agent

A LangGraph-based deep research agent that plans queries, searches the web via Tavily, and generates structured, source-grounded research reports. Includes a lightweight evaluation suite with both heuristic checks and LLM-as-a-judge scoring.

## Overview

This project implements a deep research agent that:

- Accepts a natural-language query from the user  
- Plans and executes web searches using Tavily  
- Uses LangGraph to coordinate tool usage and reasoning steps  
- Generates a structured, multi-section research report grounded in retrieved sources  
- Provides evaluation scripts to assess structure, grounding, and quality using both heuristics and an LLM judge  

The agent runs entirely in the command line and requires no frontend.

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
- Evaluation suite including:  
  - Structure checks  
  - Grounding checks  
  - LLM-as-a-Judge (1–5 scoring rubric)  
- Support for multiple research queries per session  

## Requirements

- Python 3.12+  
- OPENAI_API_KEY  
- TAVILY_API_KEY  

Install dependencies:

```bash
pip install -r requirements.txt
```  <!-- THIS WAS MISSING — now your block closes correctly -->
```
## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ai-deep-research-agent.git
cd ai-deep-research-agent
```

### 2. Create a `.env` file in the project root
```ini
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 3. (Optional but recommended) Create and activate a virtual environment

Create the venv:

```bash

python -m venv .venv

```


Activate it (Windows):
```bash
.\.venv\Scripts\activate


Activate it (Mac/Linux):

source .venv/bin/activate

```

### 4. Install dependencies
```bash
pip install -r requirements.txt

```

### Usage
Run the deep research agent
```bash
python deep_research.py
```

## You will be prompted:

Enter your research query:

```ini
Example queries:

What are the biggest safety challenges facing autonomous vehicles?

How is GenAI transforming early-stage startups?

What are the environmental risks of large-scale data centers?

How do LLMs impact cybersecurity in enterprise systems?

The agent will output a full structured report.

Run the evaluation tool
python evals.py


Evaluations produce:

A generated research report

Structure and grounding checks

LLM-as-a-judge scoring (1–5 with justification)

Ability to run multiple eval queries in one session
```

### Project Structure
ai-deep-research-agent/
│
├── deep_research.py        # LangGraph agent and report generator
├── evals.py                # Evaluation suite (heuristics + LLM judge)
├── requirements.txt        # Dependencies
├── .env                    # API keys (ignored by Git)
├── .gitignore              # Keeps venv and secrets out of GitHub
└── README.md               # Project documentation

### Notes

The .env file is ignored via .gitignore.

No secrets or virtual environment folders are tracked.

All code is written to be readable, transparent, and simple to run.

Optional Extensions

Configurable number of search calls

Configurable report structure

Support for other search APIs (SerpAPI, Exa, etc.)

Support for different LLMs

Add streaming or step-by-step reasoning outputs

License

MIT License.
