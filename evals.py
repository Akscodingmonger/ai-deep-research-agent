import re
import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from deep_research import run_research


def eval_structure(report: str) -> Dict[str, bool]:
    has_exec = "1. Executive Summary" in report
    has_key = "2. Key Findings" in report
    has_analysis = "3. Analysis" in report or "3. Analysis / Discussion" in report
    has_limits = "4. Limitations" in report
    has_sources = "5. Sources" in report
    has_all_sections = all([has_exec, has_key, has_analysis, has_limits, has_sources])
    has_citations = bool(re.search(r"\[\d+\]", report))
    has_urls_in_sources = "5. Sources" in report and "http" in report.split("5. Sources", 1)[-1]
    return {
        "has_all_sections": has_all_sections,
        "has_citations": has_citations,
        "has_urls_in_sources": has_urls_in_sources,
    }


def eval_grounding(report: str, query: str) -> Dict[str, bool]:
    tokens = re.findall(r"[a-zA-Z]+", query.lower())
    keywords = {t for t in tokens if len(t) > 3}
    lowered = report.lower()
    mentions_query_keywords = any(k in lowered for k in keywords) if keywords else True
    has_sources_section = "5. Sources" in report
    has_http_in_sources = "5. Sources" in report and "http" in report.split("5. Sources", 1)[-1]
    return {
        "mentions_query_keywords": mentions_query_keywords,
        "has_sources_section": has_sources_section,
        "has_http_in_sources": has_http_in_sources,
    }


judge_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.0,
)


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge output.")
    return json.loads(match.group(0))


def llm_judge_report(report: str, query: str) -> Dict[str, Any]:
    prompt = f"""
You are evaluating a research report produced by an AI agent.

User query:
\"\"\"{query}\"\"\"


Report:
\"\"\"{report}\"\"\"


Follow these steps internally:
1. Determine how well the report answers the user query.
2. Check whether statements appear grounded in cited sources.
3. Assess depth of analysis beyond surface-level summary.
4. Evaluate clarity, readability, and structure.

Then output only a JSON object, with no extra text, using this schema:

{{
  "relevance": <int 1-5>,
  "grounding": <int 1-5>,
  "analysis_depth": <int 1-5>,
  "clarity": <int 1-5>,
  "justification": "<2-4 concise sentences>"
}}
"""

    resp = judge_llm.invoke(prompt)
    try:
        data = _extract_json(resp.content)

        relevance = float(data.get("relevance", 1))
        grounding = float(data.get("grounding", 1))
        analysis_depth = float(data.get("analysis_depth", 1))
        clarity = float(data.get("clarity", 1))

        overall = (relevance + grounding + analysis_depth + clarity) / 4.0

        data["relevance"] = relevance
        data["grounding"] = grounding
        data["analysis_depth"] = analysis_depth
        data["clarity"] = clarity
        data["overall"] = overall

        if "justification" not in data:
            data["justification"] = "No justification provided by judge."
    except Exception:
        data = {
            "relevance": 1.0,
            "grounding": 1.0,
            "analysis_depth": 1.0,
            "clarity": 1.0,
            "overall": 1.0,
            "justification": "Failed to parse judge output as JSON.",
        }
    return data


def run_all_evals(query: str) -> Dict[str, Any]:
    report = run_research(query)
    structure = eval_structure(report)
    grounding = eval_grounding(report, query)
    llm_scores = llm_judge_report(report, query)
    return {
        "query": query,
        "report": report,
        "structure_checks": structure,
        "grounding_checks": grounding,
        "llm_judge": llm_scores,
    }


if __name__ == "__main__":
    print("Enter queries to evaluate (press Enter on an empty line or type 'exit' to quit).")
    while True:
        user_query = input("\nQuery: ").strip()
        if not user_query or user_query.lower() in {"exit", "quit"}:
            break

        results = run_all_evals(user_query)

        print("\n=== Generated Report ===\n")
        print(results["report"])

        print("\n=== Heuristic Checks ===")
        for k, v in results["structure_checks"].items():
            print(f"{k}: {v}")
        for k, v in results["grounding_checks"].items():
            print(f"{k}: {v}")

        print("\n=== LLM-as-a-Judge Scores (scale: 1 = very poor, 5 = excellent) ===")
        for k, v in results["llm_judge"].items():
            print(f"{k}: {v}")

        print("\n" + "=" * 80)
