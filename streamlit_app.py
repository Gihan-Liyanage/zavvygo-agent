import streamlit as st
import json
from agent import zavvy_handler, plan_trip_tool  # zavvy_handler is the main handler

st.set_page_config(page_title="Zavvy — Sri Lanka Travel Assistant", layout="wide")

st.title("🐢 Zavvy — Your Sri Lanka Travel Companion 🇱🇰")
st.caption("Ask Zavvy about destinations, beaches, culture, or ask it to plan a trip!")

# Sidebar: Trip planning form (calls plan_trip_tool directly for convenience)
with st.sidebar.form("trip_form"):
    st.header("✈️ Quick Plan a Trip")
    days = st.number_input("Duration (days)", min_value=1, max_value=14, value=3)
    theme = st.selectbox("Theme", ["beaches", "heritage", "wildlife", "mixed"])
    prefs = st.text_input("Preferences (comma-separated)", "relaxing, snorkeling, scenic views")
    submit_plan = st.form_submit_button("Create Itinerary")

if submit_plan:
    with st.spinner("Planning your Sri Lanka trip..."):
        req = {"theme": theme, "days": int(days), "preferences": [p.strip() for p in prefs.split(",") if p.strip()]}
        itinerary = plan_trip_tool(req)
        # show both structured and friendly summary
        st.subheader("🗓️ Itinerary (structured)")
        st.json(itinerary)
        st.subheader("✍️ Itinerary (friendly summary)")
        # Use handler to create friendly summary (we can pass a "plan" style query)
        # But zavvy_handler expects natural text, so create a small plan-invocation via choose-tool pattern by simulating user text:
        summary = None
        try:
            summary_res = zavvy_handler(f"Please create a {days}-day {theme} trip with preferences {prefs}")
            parsed = json.loads(summary_res)
            if parsed.get("type") == "plan_result":
                summary = parsed["answer"]
            else:
                summary = "Could not generate friendly summary."
        except Exception:
            summary = "Could not generate friendly summary."
        st.markdown(summary)

# --- Chat area ---
st.subheader("💬 Chat with Zavvy")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render messages
for msg in st.session_state.messages:
    role = "🧑‍💼 You" if msg["role"] == "user" else "🤖 Zavvy"
    st.markdown(f"**{role}:** {msg['text']}")

# Input
user_input = st.text_input("Type your question (e.g., 'Best surfing beaches in Sri Lanka?')", key="chat_input")

if st.button("Send"):
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "text": user_input})
        with st.spinner("Zavvy is thinking..."):
            raw = zavvy_handler(user_input)
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"type": "error", "answer": raw}
            # Choose displayed text based on type
            if parsed.get("type") == "info":
                text = parsed["answer"]
            elif parsed.get("type") == "plan_result":
                text = parsed["answer"]
            elif parsed.get("type") == "small_talk":
                text = parsed["answer"]
            elif parsed.get("type") == "self":
                text = parsed["answer"]
            else:
                text = parsed.get("answer", str(parsed))
        st.session_state.messages.append({"role": "assistant", "text": text})
        # rerun to show new messages and clear input
        st.experimental_rerun()
