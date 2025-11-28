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

### LangGraph workflow encapsulating:
- Research planning  
- Web search retrieval  
- Report generation  

### Structured output containing:
1. Executive Summary  
2. Key Findings  
3. Analysis / Discussion  
4. Limitations & Open Questions  
5. Sources (with URLs)

### Additional Capabilities
- Automatic citation indexing (e.g., [1], [2], [3])  
- Evaluation suite with:  
  - Structure checks  
  - Grounding checks  
  - LLM-as-a-Judge (1–5 scoring rubric)  
- Ability to run multiple research queries in a single session  

## Requirements

- Python 3.12+  
- Tavily API key  
- OpenAI API key  

Install dependencies:

```bash
pip install -r requirements.txt
