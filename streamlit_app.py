"""
MTS_NLP — Streamlit frontend for the Hinglish AI Summarizer.

This is only the UI. It sends the conversation to the FastAPI backend
(api.py), which hosts the mT5 model (downloaded from Google Drive), and
shows the English summary it returns.

Run (with the API already running):
    streamlit run streamlit_app.py

Config:
    API_URL env var, or the "API URL" field in the sidebar.
    Default: http://127.0.0.1:8000
"""

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

EXAMPLES = {
    "Client meeting (from the original app)": (
        "Amit: Bhai, kal dopehar apna final client meeting hai. Sab ready hai kya?\n"
        "Rahul: Database connect nahi ho raha. Main check karta hu."
    ),
    "Exam preparation": (
        "Priya: Kal maths ka paper hai aur mera calculus abhi bhi weak hai.\n"
        "Sneha: Tension mat lo, main raat ko important theorems revise karne mein help kar dungi.\n"
        "Priya: Thanks yaar, main bhi purane question papers solve karke aati hu."
    ),
    "Train travel plan": (
        "Rohan: Hum Delhi se train pakad rahe hain kal subah 6 baje wali.\n"
        "Vikram: Station pe 5 baje tak pahunch jana, traffic bohot hota hai us time.\n"
        "Rohan: Thik hai, main cab book kar dunga aaj raat ko hi."
    ),
}

st.set_page_config(page_title="Hinglish AI Summarizer", page_icon="💬", layout="centered")

st.title("💬 Hinglish AI Summarizer")
st.write(
    "Paste a **Hinglish conversation** (Hindi in Roman script, mixed with English) "
    "and get a clean **English summary**. Powered by a fine-tuned mT5 model "
    "served by the FastAPI backend — the model itself lives on Google Drive."
)

# ---------------------------------------------------------------------------
# Sidebar: API connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")

    if st.button("Check API status"):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            r.raise_for_status()
            info = r.json()
            if info.get("model_loaded"):
                st.success(f"API is up ✅  •  model loaded on {info.get('device', '?')}")
            else:
                st.warning("API is up, but the model is still loading…")
        except requests.RequestException:
            st.error("Can't reach the API. Start it first:\n`uvicorn api:app --port 8000`")

    st.divider()
    st.caption(
        "How it fits together:\n"
        "1. **Streamlit** (this page) collects the conversation\n"
        "2. **FastAPI** (`api.py`) runs the mT5 model\n"
        "3. The **model** is downloaded from **Google Drive** by the API on first start"
    )

# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------
example_choice = st.selectbox("Try an example:", ["(blank)"] + list(EXAMPLES.keys()))
default_text = EXAMPLES.get(example_choice, "")

conversation = st.text_area(
    "Hinglish conversation",
    value=default_text,
    height=200,
    placeholder="Name: message — one line per message…",
)

if st.button("✨ Generate Summary", type="primary", use_container_width=True):
    if not conversation.strip():
        st.warning("Please enter a conversation first.")
    else:
        with st.spinner("Summarizing… (first request after API start can be slow)"):
            try:
                response = requests.post(
                    f"{api_url}/api/v1/process-chat",
                    json={"conversation": conversation},
                    timeout=180,
                )
            except requests.RequestException:
                st.error(
                    "Couldn't connect to the API. Make sure it's running "
                    f"(`uvicorn api:app --port 8000`) and the API URL in the sidebar is right — "
                    f"currently `{api_url}`."
                )
                st.stop()

        if response.ok:
            data = response.json()
            st.subheader("📝 English Summary")
            st.success(data.get("summary", ""))
            if data.get("truncated"):
                st.info(
                    "Note: your conversation was longer than the model's 256-token input limit, "
                    "so only the first part was summarized."
                )
        elif response.status_code == 400:
            st.warning(response.json().get("detail", "Invalid input."))
        elif response.status_code == 503:
            st.warning("The model is still loading on the server. Wait a few seconds and try again.")
        else:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                pass
            st.error(f"API error ({response.status_code}). {detail}")
