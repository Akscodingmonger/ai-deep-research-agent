from dataclasses import dataclass
import argparse
import sys
from typing import TypedDict, Annotated
from operator import add

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_NUM_SEARCHES = 4

DEFAULT_SYSTEM_PROMPT = """
You are a meticulous research assistant. 
Using ONLY the information in the previous messages (especially the [Search Result ...] messages), 
write a structured report answering the user's query.

Required structure:
1. Executive Summary (3–5 bullet points)
2. Key Findings (with inline citations like [1], [2])
3. Analysis / Discussion
4. Limitations & Open Questions
5. Sources (numbered list with titles and URLs)

If something is uncertain or speculative, say so.
""".strip()


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    num_searches: int = DEFAULT_NUM_SEARCHES
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--num-searches", type=int, default=DEFAULT_NUM_SEARCHES)
    parser.add_argument("--system-prompt", type=str, default=None)
    return parser.parse_args()


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    query: str
    config: AgentConfig


def planner_node(state: ResearchState) -> ResearchState:
    llm = ChatOpenAI(model=state["config"].model, temperature=0.2)
    system_msg = SystemMessage(
        content=(
            "You are a research planner. Given a user query, break it into 3–6 "
            "specific web search questions. Return them as a numbered list."
        )
    )
    response = llm.invoke([system_msg, HumanMessage(content=state["query"])])
    return {**state, "messages": state["messages"] + [response]}


def web_search_node(state: ResearchState) -> ResearchState:
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    text = last_ai.content

    subquestions = [
        line.split(".", 1)[1].strip()
        for line in text.splitlines()
        if line.strip() and line.strip()[0].isdigit() and "." in line
    ] or [text]

    messages = list(state["messages"])

    tavily_search = TavilySearch(
        max_results=state["config"].num_searches,
        topic="general",
        include_answer=True,
        include_raw_content=False,
    )

    for i, q in enumerate(subquestions[: state["config"].num_searches]):
        results = tavily_search.invoke({"query": q})
        messages.append(
            AIMessage(
                content=f"[Search Result {i+1}]\nQuery: {q}\nResults:\n{results}"
            )
        )

    return {**state, "messages": messages}


def writer_node(state: ResearchState) -> ResearchState:
    llm = ChatOpenAI(model=state["config"].model, temperature=0.2)

    system_msg = SystemMessage(content=state["config"].system_prompt or DEFAULT_SYSTEM_PROMPT)
    final_prompt = HumanMessage(
        content=f"Original user query:\n{state['query']}\n\nWrite the final report now."
    )

    full_context = [system_msg] + state["messages"] + [final_prompt]
    report = llm.invoke(full_context)

    return {**state, "messages": state["messages"] + [report]}


def build_research_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("writer", writer_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "web_search")
    workflow.add_edge("web_search", "writer")
    workflow.add_edge("writer", END)
    return workflow.compile()


def run_research(query: str, config: AgentConfig) -> str:
    app = build_research_graph()

    initial_state: ResearchState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "config": config,
    }

    final_state = app.invoke(initial_state)

    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage):
            return m.content
    return "No report generated."


if __name__ == "__main__":
    args = parse_args()
    config = AgentConfig(
        model=args.model,
        num_searches=args.num_searches,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
    )

    print(f"\nUsing model={config.model}, num_searches={config.num_searches}\n")

    while True:
        query = input("Enter your research query (or 'exit'): ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        print("\nGenerating report...\n")
        result = run_research(query, config)
        print(result)
        print("\n")
