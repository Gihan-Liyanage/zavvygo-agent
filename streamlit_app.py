import streamlit as st
import json
from agent import zavvy_handler, plan_trip_tool  # zavvy_handler is the main handler

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Zavvy — Sri Lanka Travel Assistant", layout="wide")
st.title("🐢 Zavvy — Your Sri Lanka Travel Companion 🇱🇰")
st.caption("Ask Zavvy about destinations, beaches, culture, or ask it to plan a trip!")

# -----------------------------
# Sidebar: Trip Planning Form
# -----------------------------
with st.sidebar.form("trip_form"):
    st.header("✈️ Quick Plan a Trip")
    days = st.number_input("Duration (days)", min_value=1, max_value=14, value=3)
    theme = st.selectbox("Theme", ["beaches", "heritage", "wildlife", "mixed"])
    prefs = st.text_input("Preferences (comma-separated)", "relaxing, snorkeling, scenic views")
    submit_plan = st.form_submit_button("Create Itinerary")

if submit_plan:
    with st.spinner("Planning your Sri Lanka trip..."):
        # Prepare structured request
        req = {
            "theme": theme,
            "days": int(days),
            "preferences": [p.strip() for p in prefs.split(",") if p.strip()]
        }
        itinerary = plan_trip_tool(req)

        # Display structured JSON
        st.subheader("🗓️ Itinerary (structured)")
        st.json(itinerary)

        # Generate friendly summary via zavvy_handler
        st.subheader("✍️ Itinerary (friendly summary)")
        summary = None
        try:
            summary_res = zavvy_handler(
                f"Please create a {days}-day {theme} trip with preferences {prefs}"
            )
            # Try parsing JSON if returned
            if isinstance(summary_res, dict) and summary_res.get("type") == "plan_result":
                summary = summary_res["answer"]
            else:
                summary = str(summary_res)
        except Exception:
            summary = "Could not generate friendly summary."
        st.markdown(summary)

# -----------------------------
# Chat Area
# -----------------------------
st.subheader("💬 Chat with Zavvy")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "🧑‍💼 You" if msg["role"] == "user" else "🤖 Zavvy"
    st.markdown(f"**{role}:** {msg['text']}")

# Input box
user_input = st.text_input(
    "Type your question (e.g., 'Best surfing beaches in Sri Lanka?')", key="chat_input"
)

# Send button logic
if st.button("Send"):
    if user_input.strip():
        # Add user message to history
        st.session_state.messages.append({"role": "user", "text": user_input})

        with st.spinner("Zavvy is thinking..."):
            raw = zavvy_handler(user_input)

            # Ensure parsed response is dictionary
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {"type": "error", "answer": raw}
            else:
                parsed = raw

            # Determine message to display
            msg_text = parsed.get("answer", str(parsed))
            st.session_state.messages.append({"role": "assistant", "text": msg_text})

        # Rerun to refresh chat display
        st.rerun()
