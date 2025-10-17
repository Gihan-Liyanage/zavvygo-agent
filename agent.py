import os
import json
import re
from typing import Dict, Any, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

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
# State definition
# -----------------------------
class ZavvyState(TypedDict):
    user_input: str
    intent_data: Dict[str, Any]
    result: Dict[str, Any]


# -----------------------------
# Helper: Extract JSON safely
# -----------------------------
def extract_json(text: str) -> dict:
    """Extract JSON object from a string."""
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception:
        return None


# -----------------------------
# Tool: Tavily Web Search
# -----------------------------
def web_search_tool(query: str) -> Dict[str, Any]:
    """
    Performs a Tavily web search and returns structured results.
    """
    response = tavily_client.search(query=query, max_results=5)
    results = response.get("results", [])
    summary = response.get("answer", "No summary found.")
    return {"summary": summary, "results": results}


# -----------------------------
# Tool: Trip Planner
# -----------------------------
def planner_tool(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a JSON itinerary for a Sri Lanka trip based on requirements.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.3, model="gpt-4o-mini")

    days = int(requirements.get("days", 3))
    theme = requirements.get("theme", "beaches")
    preferences = requirements.get("preferences", [])

    system_prompt = (
        "You are Zavvy Planner, a travel itinerary expert specialized in Sri Lanka. "
        "Create detailed travel plans **only within Sri Lanka**."
    )

    user_prompt = (
        f"User wants a {days}-day trip in Sri Lanka focused on {theme}. "
        f"Preferences: {preferences}. "
        "Create a structured JSON itinerary with these keys:\n"
        "- trip_overview\n"
        "- days (list of Day, Activities, Hotel, Tips)\n"
        "- hotel_suggestions (list of hotels with short descriptions)\n"
        "Return only valid JSON."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    itinerary = extract_json(response.content.strip())
    if not itinerary:
        itinerary = {
            "trip_overview": f"{days}-day {theme} trip in Sri Lanka.",
            "days": [
                {
                    "day": i + 1,
                    "activities": "Could not parse full details.",
                    "hotel": "N/A",
                }
                for i in range(days)
            ],
            "hotel_suggestions": [],
        }

    return itinerary


# -----------------------------
# Node: Classify Intent
# -----------------------------
def classify_intent(state: ZavvyState) -> ZavvyState:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.2, model="gpt-4o-mini")

    user_input = state["user_input"].strip().lower()

    # Handle self-introduction queries directly
    if any(
        kw in user_input
        for kw in ["who are you", "your name", "what can you do", "introduce yourself"]
    ):
        state["intent_data"] = {"intent": "self", "query": user_input}
        return state

    decision_prompt = (
        "You are Zavvy, a travel assistant specialized in **Sri Lanka tourism**.\n"
        "Classify the user's request as:\n"
        "1. 'plan' - if they want a trip plan, itinerary, or multi-day suggestion.\n"
        "2. 'info' - if they just want factual travel info.\n\n"
        "Return JSON like {\"intent\": \"plan\", \"params\": {...}} or "
        "{\"intent\": \"info\", \"query\": \"...\"}.\n\n"
        f"User: {state['user_input']}"
    )

    decision = llm.invoke([{"role": "user", "content": decision_prompt}])

    try:
        intent_data = json.loads(decision.content)
    except Exception:
        if any(word in user_input for word in ["plan", "trip", "itinerary"]):
            intent_data = {"intent": "plan", "params": {"theme": "beaches", "days": 3}}
        else:
            intent_data = {"intent": "info", "query": state["user_input"]}

    state["intent_data"] = intent_data
    return state


# -----------------------------
# Node: Handle Info Queries
# -----------------------------
def handle_info(state: ZavvyState) -> ZavvyState:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.2, model="gpt-4o-mini")

    query = state["intent_data"].get("query", state["user_input"])
    search_data = web_search_tool(f"{query} Sri Lanka tourism")

    answer_prompt = (
        "You are Zavvy, a Sri Lanka travel expert. "
        "Summarize the following Tavily search results into a short, factual, Sri Lanka-specific answer. "
        "Answer in 3–6 sentences and cite sources as [1], [2], etc.\n\n"
        f"Summary: {search_data['summary']}\n\n"
        f"Results: {json.dumps(search_data['results'], indent=2)}"
    )

    response = llm.invoke([{"role": "user", "content": answer_prompt}])

    state["result"] = {
        "type": "info",
        "answer": response.content.strip(),
        "sources": search_data["results"],
    }
    return state


# -----------------------------
# Node: Handle Trip Planning
# -----------------------------
def handle_plan(state: ZavvyState) -> ZavvyState:
    params = state["intent_data"].get("params", {})
    itinerary = planner_tool(params)
    state["result"] = {"type": "plan_result", "plan": itinerary}
    return state


# -----------------------------
# Node: Handle Self-Queries
# -----------------------------
def handle_self(state: ZavvyState) -> ZavvyState:
    state["result"] = {
        "type": "info",
        "answer": (
            "I'm **Zavvy**, your Sri Lanka travel companion 🇱🇰. "
            "I can answer questions about destinations, beaches, heritage sites, wildlife, and culture — "
            "and help you plan beautiful trips across Sri Lanka!"
        ),
    }
    return state


# -----------------------------
# Router Function
# -----------------------------
def router(state: ZavvyState) -> str:
    intent = state["intent_data"].get("intent", "info")
    if intent == "plan":
        return "plan"
    elif intent == "self":
        return "self"
    return "info"


# -----------------------------
# Graph Definition
# -----------------------------
def build_zavvy_agent():
    graph = StateGraph(ZavvyState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("info", handle_info)
    graph.add_node("plan", handle_plan)
    graph.add_node("self", handle_self)

    graph.add_conditional_edges(
        "classify_intent", router, {"plan": "plan", "info": "info", "self": "self"}
    )

    graph.add_edge("info", END)
    graph.add_edge("plan", END)
    graph.add_edge("self", END)

    graph.set_entry_point("classify_intent")

    zavvy_graph = graph.compile()

    def zavvy_handler(user_input: str) -> str:
        output = zavvy_graph.invoke({"user_input": user_input})
        return json.dumps(output["result"], indent=2)

    return zavvy_handler, planner_tool
