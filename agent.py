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
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception:
        return None


# -----------------------------
# Tool: Tavily Web Search
# -----------------------------
def web_search_tool(query: str) -> Dict[str, Any]:
    response = tavily_client.search(query=query, max_results=5)
    results = response.get("results", [])
    summary = response.get("answer", "No summary found.")
    return {"summary": summary, "results": results}


# -----------------------------
# Tool: Trip Planner
# -----------------------------
def planner_tool(requirements: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.4, model="gpt-4o-mini")

    days = int(requirements.get("days", 3))
    theme = requirements.get("theme", "beaches")
    preferences = requirements.get("preferences", [])

    system_prompt = (
        "You are Zavvy Planner, a friendly Sri Lankan travel expert. "
        "Your tone is warm, helpful, and conversational—like a local friend giving advice. "
        "Keep things informative but easy to read. "
        "Always stay within Sri Lanka."
    )

    user_prompt = (
        f"Create a friendly, detailed {days}-day trip itinerary in Sri Lanka around the theme '{theme}'.\n"
        f"User preferences: {preferences}.\n\n"
        "Return JSON with:\n"
        "- trip_overview (friendly tone)\n"
        "- days (each day: day, activities, hotel, tips)\n"
        "- hotel_suggestions (3–6 hotels)\n"
        "Return only JSON."
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
                    "activities": "Could not parse.",
                    "hotel": "N/A",
                    "tips": ""
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

    if any(q in user_input for q in ["who are you", "your name", "introduce yourself"]):
        state["intent_data"] = {"intent": "self", "query": user_input}
        return state

    decision_prompt = (
        "You are Zavvy, a friendly Sri Lanka travel buddy. "
        "Classify user intent.\n\n"
        "Return JSON:\n"
        "- {\"intent\": \"plan\", \"params\": {...}}\n"
        "- {\"intent\": \"info\", \"query\": \"...\"}\n\n"
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
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.5, model="gpt-4o-mini")

    query = state["intent_data"].get("query", state["user_input"])
    search_data = web_search_tool(f"{query} Sri Lanka tourism")

    # ✨ FRIENDLY TONE UPDATE
    answer_prompt = (
        "You are Zavvy, a warm, friendly Sri Lankan travel guide. "
        "Use the Tavily results to craft an approachable, natural explanation (5–8 sentences). "
        "Speak like a local friend who knows the country well. "
        "Do NOT sound like a search engine. No citations. No robotic tone.\n\n"
        f"Tavily summary: {search_data['summary']}\n\n"
        f"Tavily results: {json.dumps(search_data['results'], indent=2)}"
    )

    response = llm.invoke([{"role": "user", "content": answer_prompt}])

    state["result"] = {
        "type": "info",
        "answer": response.content.strip(),
    }
    return state


# -----------------------------
# Node: Handle Self-Queries
# -----------------------------
def handle_self(state: ZavvyState) -> ZavvyState:
    state["result"] = {
        "type": "info",
        "answer": (
            "I'm Zavvy — your friendly Sri Lankan travel buddy 🇱🇰. "
            "Ask me anything about beaches, adventures, culture, food, or even help planning a full trip!"
        )
    }
    return state


# -----------------------------
# Router
# -----------------------------
def router(state: ZavvyState) -> str:
    intent = state["intent_data"].get("intent", "info")
    return intent


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
        "classify_intent",
        router,
        {"plan": "plan", "info": "info", "self": "self"},
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
