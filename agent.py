import os
import json
import re
from typing import Dict, Any, TypedDict, Optional, List
from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI

# Tavily
from tavily import TavilyClient


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not set in environment")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# -----------------------------
# Types
# -----------------------------
class ZavvyResult(TypedDict):
    type: str
    answer: Any
    meta: Optional[Dict[str, Any]]


# -----------------------------
# Memory state for LangGraph
# -----------------------------
class AgentState(MessagesState):
    """LangGraph state: holds conversation memory"""
    pass


# -----------------------------
# Helpers
# -----------------------------
def safe_extract_json(text: str) -> Optional[dict]:
    """Robust JSON extraction."""
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        pass
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except:
        pass
    return None


# -----------------------------
# Tools
# -----------------------------
def web_search_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    response = tavily_client.search(query=query, max_results=max_results)
    return {
        "summary": response.get("answer", "") or "",
        "results": response.get("results", []),
    }


def plan_trip_tool(requirements: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.35, model="gpt-4o-mini")

    days = int(requirements.get("days", 3))
    theme = requirements.get("theme", "mixed")
    preferences = requirements.get("preferences", [])

    system_prompt = (
        "You are Zavvy Planner — friendly and concise. "
        "Create a Sri Lanka itinerary in structured JSON."
    )

    user_prompt = (
        f"Create a {days}-day itinerary for theme '{theme}'.\n"
        f"Preferences: {preferences}\n"
        "Return only JSON with keys: trip_overview, days, hotel_suggestions\n"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    itinerary = safe_extract_json(response.content.strip())
    if not itinerary:
        itinerary = {
            "trip_overview": f"A {days}-day {theme} trip in Sri Lanka.",
            "days": [
                {
                    "day": i + 1,
                    "title": f"Day {i+1}",
                    "activities": ["Exploration planned."],
                    "hotel": "Available on request",
                    "tips": "Ask for local dining tips."
                }
                for i in range(days)
            ],
            "hotel_suggestions": [],
        }

    return itinerary


# -----------------------------
# Intent Routing via LLM
# -----------------------------
def choose_tool_with_llm(user_message: str) -> Dict[str, Any]:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, model="gpt-4o-mini")
    prompt = (
        "Choose which tool must handle this.\n\n"
        "Tools:\n"
        "- info\n- plan\n- small_talk\n- self\n\n"
        "Respond ONLY as JSON: {\"tool\": \"...\", \"args\": {...}}\n"
        f"USER: {user_message}"
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    parsed = safe_extract_json(response.content.strip())

    if not parsed:
        return {"tool": "info", "args": {"query": user_message}}

    tool = parsed.get("tool")
    args = parsed.get("args", {})

    if tool not in {"info", "plan", "small_talk", "self"}:
        return {"tool": "info", "args": {"query": user_message}}

    return {"tool": tool, "args": args}


# -----------------------------
# Render friendly results
# -----------------------------
def render_friendly_info(search_data: Dict[str, Any], user_message: str) -> str:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.6, model="gpt-4o-mini")

    system_prompt = (
        "You are Zavvy — friendly Sri Lankan travel buddy. "
        "Create a conversational answer (4–6 sentences). No citations."
    )

    user_prompt = (
        f"User asked: {user_message}\n\n"
        f"Tavily summary:\n{search_data.get('summary')}\n"
        f"Results:\n{json.dumps(search_data.get('results', [])[:5], indent=2)}"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response.content.strip()


def render_friendly_plan(itinerary: Dict[str, Any]) -> str:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.45, model="gpt-4o-mini")

    system_prompt = "Summarize this itinerary warmly (3–6 paragraphs)."

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(itinerary, indent=2)},
        ]
    )

    return response.content.strip()


# -----------------------------
# ORIGINAL handler logic (kept)
# -----------------------------
def zavvy_handler(user_input: str) -> ZavvyResult:
    user_input = user_input.strip()
    dispatch = choose_tool_with_llm(user_input)

    tool = dispatch["tool"]
    args = dispatch.get("args", {})

    try:
        if tool == "small_talk":
            llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7, model="gpt-4o-mini")
            reply = llm.invoke(
                [
                    {"role": "system", "content": "Friendly and brief."},
                    {"role": "user", "content": user_input},
                ]
            )
            return {"type": "small_talk", "answer": reply.content.strip(), "meta": dispatch}

        if tool == "self":
            return {
                "type": "self",
                "answer": "I'm Zavvy — your Sri Lanka travel buddy 🇱🇰.",
                "meta": dispatch,
            }

        if tool == "plan":
            days = int(args.get("days", 3))
            theme = args.get("theme", "beaches")
            preferences = args.get("preferences", [])
            itinerary = plan_trip_tool({"days": days, "theme": theme, "preferences": preferences})
            return {
                "type": "plan_result",
                "answer": render_friendly_plan(itinerary),
                "meta": {"itinerary": itinerary},
            }

        # default → info
        query = args.get("query", user_input)
        search_data = web_search_tool(f"{query} Sri Lanka tourism")
        return {
            "type": "info",
            "answer": render_friendly_info(search_data, user_input),
            "meta": {"search": search_data},
        }

    except Exception as e:
        return {
            "type": "error",
            "answer": "Sorry, I couldn’t process that now.",
            "meta": {"error": str(e)},
        }


# -----------------------------
# LangGraph Nodes
# -----------------------------
def route_node(state: AgentState):
    """Extract the latest user message and run routing logic."""
    last = state["messages"][-1].content
    tool_result = zavvy_handler(last)

    # Store inside message format that LLM will understand
    return {
        "messages": [
            {
                "role": "assistant",
                "content": tool_result["answer"],
            }
        ]
    }


# -----------------------------
# Build the graph
# -----------------------------
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.set_entry_point("route")

    return graph.compile()
