import os
import json
import re
from typing import Dict, Any, TypedDict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
# Helpers
# -----------------------------
def safe_extract_json(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from text robustly.
    Returns dict or None.
    """
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find first {...} block
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            candidate = m.group(0)
            return json.loads(candidate)
    except Exception:
        pass

    # Last resort: find lines that look like key: value and build JSON (very rare)
    return None


# -----------------------------
# Tools (pure functions)
# -----------------------------
def web_search_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Use Tavily to fetch web results and summary.
    Returns dict: { summary: str, results: list }
    """
    response = tavily_client.search(query=query, max_results=max_results)
    results = response.get("results", [])
    summary = response.get("answer", "") or ""
    return {"summary": summary, "results": results}


def plan_trip_tool(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a friendly itinerary using the LLM.
    Returns a JSON-serializable itinerary.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.35, model="gpt-4o-mini")

    days = int(requirements.get("days", 3))
    theme = requirements.get("theme", "mixed")
    preferences = requirements.get("preferences", [])

    system_prompt = (
        "You are Zavvy Planner — friendly, concise, and conversational. "
        "Create an easy-to-follow Sri Lanka itinerary. Keep tone like a helpful local friend."
        "Only produce JSON when asked for structured output, otherwise produce friendly text."
    )

    # Request structured JSON itinerary
    user_prompt = (
        f"Create a {days}-day itinerary around the theme '{theme}' for travel within Sri Lanka.\n"
        f"Preferences: {preferences}\n\n"
        "Return a JSON object with keys:\n"
        "- trip_overview (short friendly paragraph)\n"
        "- days (list of objects with: day (int), title (short), activities (list of strings), hotel (short suggestion), tips (short))\n"
        "- hotel_suggestions (list of up to 6 short hotel descriptions)\n\n"
        "Return valid JSON only."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    itinerary = safe_extract_json(response.content.strip())
    if not itinerary:
        # Safe fallback: simple structure
        itinerary = {
            "trip_overview": f"A {days}-day {theme} trip in Sri Lanka.",
            "days": [
                {
                    "day": i + 1,
                    "title": f"Day {i + 1}",
                    "activities": ["Local exploring; details will be tailored on follow up."],
                    "hotel": "Recommended on request",
                    "tips": "Ask for local dining suggestions."
                }
                for i in range(days)
            ],
            "hotel_suggestions": [],
        }

    return itinerary


# -----------------------------
# LLM-driven Tool Chooser (no hard-coded keywords)
# -----------------------------
def choose_tool_with_llm(user_message: str) -> Dict[str, Any]:
    """
    Ask the LLM to return a single JSON telling us which tool to call and arguments.
    Expected JSON schema:
    {
      "tool": "<one of: info, plan, small_talk, self>",
      "args": { ... }
    }
    The LLM must respond ONLY with JSON. We parse safely and fall back to 'info' on failure.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, model="gpt-4o-mini")
    prompt = (
        "You are an assistant that *only* chooses which tool should handle the user's message.\n\n"
        "Available tools:\n"
        "- info : use to fetch and summarize web info about a Sri Lanka place or travel question. arguments: {\"query\": \"...\"}\n"
        "- plan : use to build a trip itinerary. arguments: {\"days\": <int>, \"theme\": \"beaches|heritage|wildlife|mixed\", \"preferences\": [..]}\n"
        "- small_talk : short chitchat/greetings. arguments: {\"message\": \"...\"}\n"
        "- self : questions about the assistant. arguments: {}\n\n"
        "RULES (very strict):\n"
        "1) Respond with ONLY a single JSON object and nothing else.\n"
        "2) JSON must contain exactly keys: tool, args\n"
        "3) tool value must be one of: \"info\", \"plan\", \"small_talk\", \"self\"\n"
        "4) args must be an object (may be empty)\n\n"
        "If you are not sure, pick \"info\" and set args.query to the user's message.\n\n"
        "USER MESSAGE:\n"
        f"{user_message}\n\n"
        "Respond now with the JSON only."
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    parsed = safe_extract_json(response.content.strip())
    if not parsed:
        # fallback: treat as info query
        return {"tool": "info", "args": {"query": user_message}}
    # Defensive normalization
    tool = parsed.get("tool")
    args = parsed.get("args", {})
    if tool not in {"info", "plan", "small_talk", "self"} or not isinstance(args, dict):
        return {"tool": "info", "args": {"query": user_message}}
    return {"tool": tool, "args": args}


# -----------------------------
# User-facing formatter (makes final text friendly)
# -----------------------------
def render_friendly_info(search_data: Dict[str, Any], user_message: str) -> str:
    """
    Ask the LLM to convert search_data into a friendly answer.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.6, model="gpt-4o-mini")
    system_prompt = (
        "You are Zavvy — a warm, friendly Sri Lankan travel companion. "
        "Turn the provided search summary and results into a short (4-6 sentence) "
        "friendly, conversational answer tailored to the user's message. "
        "Do NOT output citations. Keep it helpful and local-sounding."
    )

    user_prompt = (
        f"User asked: {user_message}\n\n"
        f"Tavily summary:\n{search_data.get('summary','')}\n\n"
        f"Tavily results (top items):\n{json.dumps(search_data.get('results', [])[:5], indent=2)}\n\n"
        "Write a friendly answer now."
    )

    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    return response.content.strip()


def render_friendly_plan(itinerary: Dict[str, Any]) -> str:
    """
    Create a short friendly textual summary of a structured itinerary.
    """
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.45, model="gpt-4o-mini")
    system_prompt = "You are Zavvy Planner: friendly, concise. Summarize the JSON itinerary into a warm, human-friendly description (3-6 short paragraphs)."

    user_prompt = f"Here is an itinerary JSON:\n{json.dumps(itinerary, indent=2)}\n\nPlease summarize it into friendly text for the user."

    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    return response.content.strip()


# -----------------------------
# Main handler (public)
# -----------------------------
def zavvy_handler(user_input: str) -> str:
    """
    Main single-entry handler. Returns a JSON string (ZavvyResult).
    The flow:
      - ask LLM which tool to call (JSON-only)
      - call tool
      - if required, ask LLM to produce final friendly answer
    """
    user_input = user_input.strip()
    dispatch = choose_tool_with_llm(user_input)
    tool = dispatch["tool"]
    args = dispatch.get("args", {})

    # Default response structure
    result: ZavvyResult = {"type": "info", "answer": "Sorry — something went wrong.", "meta": {"tool": tool, "args": args}}

    try:
        if tool == "small_talk":
            # short friendly reply (do not call external search)
            # Let LLM create a short reply based on message
            llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7, model="gpt-4o-mini")
            prompt = (
                "You are Zavvy — friendly and brief. Reply to the user message with a short warm reply (1-3 sentences). "
                "Be conversational and concise."
            )
            response = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": user_input}])
            result = {"type": "small_talk", "answer": response.content.strip(), "meta": {"tool": "small_talk"}}

        elif tool == "self":
            # Describe assistant capabilities
            answer = (
                "I'm Zavvy — your friendly Sri Lanka travel companion 🇱🇰. "
                "I can help with destination info, local tips, and building trip itineraries."
            )
            result = {"type": "self", "answer": answer, "meta": {"tool": "self"}}

        elif tool == "plan":
            # Expect args to contain days/theme/preferences; fallback defaults
            days = int(args.get("days", 3))
            theme = args.get("theme", "beaches")
            preferences = args.get("preferences", [])
            req = {"days": days, "theme": theme, "preferences": preferences}
            itinerary = plan_trip_tool(req)
            friendly = render_friendly_plan(itinerary)
            result = {"type": "plan_result", "answer": friendly, "meta": {"itinerary": itinerary}}

        else:  # info or fallback to info
            query = args.get("query", user_input)
            search_data = web_search_tool(f"{query} Sri Lanka tourism")
            friendly = render_friendly_info(search_data, user_input)
            result = {"type": "info", "answer": friendly, "meta": {"search": search_data}}

    except Exception as e:
        # graceful fallback
        result = {
            "type": "error",
            "answer": "Sorry, I couldn't process that right now. Could you try rephrasing?",
            "meta": {"error": str(e)},
        }

    return json.dumps(result, indent=2)

