import streamlit as st
import json
from agent import build_zavvy_agent

st.set_page_config(page_title="Zavvy — Sri Lanka Travel Assistant", layout="wide")

zavvy_handler, planner_tool = build_zavvy_agent()

st.title("🐢 Zavvy — Your Sri Lanka Travel Companion 🇱🇰")
st.caption("Ask Zavvy about destinations, beaches, cultural sites, or ask to plan a Sri Lankan trip!")

with st.sidebar.form("trip_form"):
    st.header("✈️ Plan a Trip")
    days = st.number_input("Duration (days)", min_value=1, max_value=14, value=3)
    theme = st.selectbox("Theme", ["beaches", "heritage", "wildlife", "mixed"])
    prefs = st.text_input("Preferences (comma-separated)", "relaxing, snorkeling, scenic views")
    submit_plan = st.form_submit_button("Create Itinerary")

if submit_plan:
    with st.spinner("Planning your Sri Lanka trip..."):
        req = {"theme": theme, "days": days, "preferences": [p.strip() for p in prefs.split(",") if p.strip()]}
        plan = planner_tool(req)
    st.subheader("🗓️ Your Itinerary")
    st.json(plan)

st.subheader("💬 Chat with Zavvy")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "🧑‍💼 You" if msg["role"] == "user" else "🤖 Zavvy"
    st.markdown(f"**{role}:** {msg['text']}")

user_input = st.text_input("Type your question…", key="chat_input")

if st.button("Send"):
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "text": user_input})
        with st.spinner("Zavvy is thinking..."):
            response = zavvy_handler(user_input)

        try:
            parsed = json.loads(response)
            if parsed.get("type") == "info":
                text = parsed["answer"]
            elif parsed.get("type") == "plan_result":
                text = f"Here's your plan:\n\n```json\n{json.dumps(parsed['plan'], indent=2)}\n```"
            else:
                text = response
        except Exception:
            text = response

        st.session_state.messages.append({"role": "assistant", "text": text})
        st.rerun()
