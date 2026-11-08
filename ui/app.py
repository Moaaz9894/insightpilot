"""Chat UI.  streamlit run ui/app.py   (set API_URL / API_KEY if the API is remote)"""
import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
HEADERS = {"x-api-key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}

st.set_page_config(page_title="InsightPilot", page_icon="📈")
st.title("📈 InsightPilot")
st.caption("Ask about sales, forecasts and company policies.")

if "msgs" not in st.session_state:
    st.session_state.msgs = []
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("e.g. Forecast SKU-001 in North for 14 days"):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            r = requests.post(f"{API}/chat", json={"question": q, "history": st.session_state.msgs[:-1][-6:]},
                              headers=HEADERS, timeout=60)
            r.raise_for_status()
            d = r.json()
            st.markdown(d["answer"])
            with st.expander(f"Details · {d['mode']} · {d['latency_ms']} ms"):
                st.json(d["tool_calls"])
            st.session_state.msgs.append({"role": "assistant", "content": d["answer"]})
        except Exception as e:
            st.error(f"API error: {e}")
