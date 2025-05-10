# ─── Cloud‑only SQLite shim ────────────────────────────────────────────────────
__import__("pysqlite3")          # ⚠️ Comment these three lines if running locally
import sys                       # ⚠️ Comment these three lines if running locally
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # ⚠️ Comment these if local

# ─── Standard libs ─────────────────────────────────────────────────────────────
import os, uuid, json, tempfile, shutil, warnings
from datetime import datetime

# ─── Third‑party libs ──────────────────────────────────────────────────────────
import streamlit as st
import chromadb
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning)
chromadb.api.client.SharedSystemClient.clear_system_cache()

# ────────────────────── PATCH for pydantic “class‑not‑fully‑defined” ──────────
"""
Some Streamlit‑Cloud containers load Pydantic‑v2 and LangChain in an order that
makes forward‑references inside `HuggingFaceHub` unavailable at import time.
We provide any missing symbols *then* call `model_rebuild(force=True)` so
Pydantic re‑parses the model with everything present.
"""
from typing import Union  # noqa: F401

# Provide / stub‑out BaseCache if LangChain didn't export it yet
try:
    from langchain_core.cache import BaseCache  # LangChain ≥ 0.2
except ImportError:
    try:
        from langchain.cache import BaseCache   # Older LangChain
    except ImportError:
        # Fallback dummy to satisfy the type reference
        class BaseCache:  # type: ignore
            """Minimal stand‑in to satisfy forward refs."""
            pass

import sys as _sys

# Inject BaseCache into HuggingFaceHub's module if still missing
_hf_mod = _sys.modules[HuggingFaceHub.__module__]
if not hasattr(_hf_mod, "BaseCache"):
    setattr(_hf_mod, "BaseCache", BaseCache)

# Finally, rebuild the model so all annotations resolve
try:
    HuggingFaceHub.model_rebuild(force=True)
except Exception:
    # If rebuild fails we continue; a second failure will raise below
    pass
# ───────────────────────────────────────────────────────────────────────────────

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Webapp",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# ─── Hugging Face embedding model (BAAI/bge‑base) ─────────────────────────────
HF_TOKEN = st.secrets["API_TOKEN"]          # put your token in `.streamlit/secrets.toml`
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)

# ─── Persona‑specific prompt templates ─────────────────────────────────────────
PERSONA_TEMPLATES = {
    "Friendly": """Answer the question in a warm, conversational tone based ONLY on the following context:
    {context}

    Question: {question}

    If you don't know the answer, just say that you don't know. Keep it friendly and approachable!
    """,
    "Formal": """Answer the question in a professional and respectful tone based ONLY on the following context:
    {context}

    Question: {question}

    If you don't know the
