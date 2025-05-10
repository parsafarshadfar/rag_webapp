"""
Streamlit RAG chat‑with‑PDF app (rev. 2025‑05‑10)
-------------------------------------------------
Key fixes
~~~~~~~~~
*   Robust handling of HuggingFace rate‑limits – `llm` is *always* defined or
    the app stops gracefully.
*   `retry_with_proxies()` now returns **an LLM object** rather than a string
    response.
*   Catches concrete `HTTPError` and inspects status codes instead of fragile
    substring checks.
*   Tidied duplicates and removed unused imports.

Tested with the pinned versions in *requirements.txt* (Python 3.12).
"""

# ---------------------------------------------------------------------------
# Monkey‑patch for pysqlite3 on Streamlit Community Cloud
# ---------------------------------------------------------------------------
__import__("pysqlite3")                     # noqa: E402 – must run *very* early
import sys as _sys                          # (community cloud lacks system SQLite)
_sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import os
import uuid
import shutil
import json
import time
import tempfile
import warnings
from datetime import datetime

# Third‑party imports
import requests
from requests.exceptions import HTTPError
import streamlit as st

import chromadb
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser

# ----------------------------------------------------------------------------
# Streamlit & global config
# ----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(
    page_title="RAG Webapp",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
print("Retrieving…")

# Clear any leftover Chroma cache in the shared Community Cloud runtime
chromadb.api.client.SharedSystemClient.clear_system_cache()

# ----------------------------------------------------------------------------
# Hugging Face & embeddings
# ----------------------------------------------------------------------------
HF_TOKEN = st.secrets["API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="BAAI/bge-base-en-v1.5",
)

# ----------------------------------------------------------------------------
# Helper functions – proxy handling
# ----------------------------------------------------------------------------

def fetch_free_proxies() -> list[dict]:
    """Fetch a bunch of free HTTPS proxies from public lists."""
    proxy_sources = [
        "https://www.proxy-list.download/api/v1/get?type=https",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
        "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/all/data.txt",
    ]
    proxies: list[dict] = []
    for url in proxy_sources:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                for p in r.text.strip().split("\n"):
                    proxies.append({"http": f"http://{p}", "https": f"http://{p}"})
        except Exception as e:  # pragma: no cover – best‑effort only
            st.warning(f"Proxy‑list fetch failed: {e}")
    return proxies


def test_proxy(proxy: dict) -> bool:
    """Return *True* if the proxy can reach HuggingFace quickly enough."""
    TEST_URL = "https://api-inference.huggingface.co/models"
    try:
        t0 = time.time()
        resp = requests.head(TEST_URL, proxies=proxy, timeout=2)
        return resp.status_code == 200 and (time.time() - t0) < 2
    except Exception:
        return False


def retry_with_proxies(repo_id: str, model_kwargs: dict):
    """Try each free proxy until we get a working LLM object or exhaust the list."""
    for proxy in fetch_free_proxies():
        if not test_proxy(proxy):
            continue
        try:
            return HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs, proxies=proxy)
        except HTTPError as e:
            if e.response is None or e.response.status_code != 429:
                # Different error – don’t silently swallow it
                raise
            # else: rate‑limit again → try next proxy
    return None

# ----------------------------------------------------------------------------
# LLM initialisation – *always* end up with a usable object or stop the app
# ----------------------------------------------------------------------------
repo_id = "huggingfaceh4/zephyr-7b-alpha"
model_kwargs = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": st.session_state.get("temperature_value", 0.5),
    "top_p": 0.9,
    "return_full_text": False,
}

llm = None
try:
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
except HTTPError as e:
    if e.response is not None and e.response.status_code == 429:
        st.info("HF rate‑limit hit – retrying with free proxies …")
        llm = retry_with_proxies(repo_id, model_kwargs)
    else:
        st.error(f"LLM initialisation failed: {e}")
        st.stop()
except Exception as e:
    st.error(f"Unexpected error while creating the LLM: {e}")
    st.stop()

if llm is None:
    st.error("Could not create a HuggingFaceHub LLM (all proxies exhausted).")
    st.stop()

# ----------------------------------------------------------------------------
# Vector‑store & retriever (one collection per browser session)
# ----------------------------------------------------------------------------
SESSION_ID = str(uuid.uuid4())
st.session_state["vector_db"] = Chroma(
    embedding_function=embeddings,
    collection_name=SESSION_ID,
)
retriever = st.session_state["vector_db"].as_retriever()

# ----------------------------------------------------------------------------
# Prompt personas
# ----------------------------------------------------------------------------
PERSONA_TEMPLATES = {
    "Friendly": """Answer the question in a warm, conversational tone based **only** on the following context:\n{context}\n\nQuestion: {question}\n\nIf you don't know the answer, just say that you don't know. Keep it friendly and approachable!""",
    "Formal": """Answer the question in a professional and respectful tone based **only** on the following context:\n{context}\n\nQuestion: {question}\n\nIf you don't know the answer, state it politely without making up any information.""",
    "Technical": """Answer the question in a technical and detailed manner based **only** on the following context:\n{context}\n\nQuestion: {question}\n\nProvide accurate, in‑depth answers. Do **not** guess if the answer is not in the context.""",
    "Concise": """Answer the question briefly and to the point based **only** on the following context:\n{context}\n\nQuestion: {question}\n\nKeep responses short and straightforward.""",
}

# ----------------------------------------------------------------------------
# Chain (RAG pipeline) – built dynamically based on the selected persona
# ----------------------------------------------------------------------------

def build_chain(persona: str):
    prompt = ChatPromptTemplate.from_template(PERSONA_TEMPLATES[persona])
    return (
        {
            "context": retriever.with_config(run_name="Docs"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# ----------------------------------------------------------------------------
# Sidebar – file upload, temperature, persona, housekeeping
# ----------------------------------------------------------------------------
file_upload = st.sidebar.file_uploader("Upload a PDF", type="pdf")

st.session_state["temperature_value"] = st.sidebar.slider(
    "LLM temperature", 0.05, 1.0, 0.5, step=0.05
)
st.sidebar.write("Lower values → answers stick closer to the document text.")

st.session_state["persona"] = st.sidebar.selectbox(
    "Assistant tone", list(PERSONA_TEMPLATES.keys()), index=2
)

if st.sidebar.button("Delete vector store"):
    st.session_state.pop("vector_db", None)
    st.success("Vector DB cleared.")

# ----------------------------------------------------------------------------
# Main app title & chat history initialisation
# ----------------------------------------------------------------------------
st.title("📄🤖 Chat with your PDF")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------------------------------------------------
# Handle PDF upload – build / refresh vector DB
# ----------------------------------------------------------------------------
if file_upload:
    st.session_state["file_name"] = file_upload.name

    # Refresh vector store for the new document
    st.session_state["vector_db"].delete_collection()

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"{SESSION_ID}_{file_upload.name}")
    with open(tmp_path, "wb") as fh:
        fh.write(file_upload.getbuffer())

    docs = PyPDFLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128).split_documents(docs)

    st.session_state["vector_db"] = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=SESSION_ID,
    )
    retriever = st.session_state["vector_db"].as_retriever()
    shutil.rmtree(tmp_dir)
    st.success("PDF indexed – ask away!")

# ----------------------------------------------------------------------------
# Chat UI
# ----------------------------------------------------------------------------
if file_upload:
    chain = build_chain(st.session_state["persona"])  # persona‑specific prompt

    container = st.container(height=600, border=True)
    for msg in st.session_state.chat_history:
        avatar = "🤖" if msg["role"] == "assistant" else "🤔"
        with container.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type your question …"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        container.chat_message("user", avatar="🤔").markdown(user_input)

        with container.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking …"):
                try:
                    answer = chain.invoke(user_input)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    raise
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

# ----------------------------------------------------------------------------
# Chat‑history download
# ----------------------------------------------------------------------------
if st.session_state.chat_history:
    if st.sidebar.download_button(
        "Download chat as JSON",
        json.dumps(st.session_state.chat_history, indent=2),
        file_name=f"History_{st.session_state.get('file_name', 'session')}_{datetime.now():%Y%m%d}.json",
        mime="application/json",
    ):
        st.balloons()
