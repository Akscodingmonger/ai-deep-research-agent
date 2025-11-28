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


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    query: str


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
)

tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True,
    include_raw_content=False,
)


def planner_node(state: ResearchState) -> ResearchState:
    user_query = state["query"]

    system_msg = SystemMessage(
        content=(
            "You are a research planner. Given a user query, break it into 3–6 "
            "concrete, focused web search questions. Return them as a numbered "
            "list, with each line starting with '1.', '2.', etc."
        )
    )

    response = llm.invoke([system_msg, HumanMessage(content=user_query)])

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

    for i, q in enumerate(subquestions[:4]):
        results = tavily_search.invoke({"query": q})
        messages.append(
            AIMessage(
                content=f"[Search Result {i+1}]\nQuery: {q}\nResults:\n{results}"
            )
        )

    return {**state, "messages": messages}


def writer_node(state: ResearchState) -> ResearchState:
    system_msg = SystemMessage(
        content=(
            "You are a meticulous research assistant. Using ONLY the information in the "
            "previous messages (especially the [Search Result ...] messages), write a "
            "structured report answering the user's query.\n\n"
            "Required structure:\n"
            "1. Executive Summary (3–5 bullet points)\n"
            "2. Key Findings (with inline citations like [1], [2])\n"
            "3. Analysis / Discussion\n"
            "4. Limitations & Open Questions\n"
            "5. Sources (numbered list with titles and URLs)\n\n"
            "If something is uncertain or speculative, say so explicitly."
        )
    )

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


def run_research(query: str) -> str:
    app = build_research_graph()

    initial_state: ResearchState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
    }

    final_state = app.invoke(initial_state)

    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage):
            return m.content

    return "No report generated."


if __name__ == "__main__":
    graph = build_research_graph()

    while True:
        query = input("Enter your research query (or 'exit'): ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        initial_state = {"messages": [], "query": query}
        final_state = graph.invoke(initial_state)

        last_ai = next(m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage))
        print(last_ai.content)
        print()
