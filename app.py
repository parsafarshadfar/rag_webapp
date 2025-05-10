import streamlit as st

# ─── MUST be the first Streamlit command ──────────────────────────────────────
st.set_page_config(
    page_title="RAG Webapp",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)
# ───────────────────────────────────────────────────────────────────────────────

# Standard‑library imports -----------------------------------------------------
__import__("pysqlite3")          # ⚠️ Comment these three lines if running locally
import sys                       # ⚠️ Comment these three lines if running locally
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # ⚠️ Comment these if local

import os, uuid, json, tempfile, shutil, warnings
from datetime import datetime
from typing import Union  # needed for pydantic forward refs

# Third‑party imports ----------------------------------------------------------
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

# ── Patch for pydantic “class‑not‑fully‑defined” in HuggingFaceHub ────────────
try:
    from langchain_core.cache import BaseCache
except ImportError:
    try:
        from langchain.cache import BaseCache
    except ImportError:
        class BaseCache:  # type: ignore
            """Minimal stub for missing BaseCache forward ref."""
            pass

_hf_mod = sys.modules[HuggingFaceHub.__module__]
if not hasattr(_hf_mod, "BaseCache"):
    setattr(_hf_mod, "BaseCache", BaseCache)

try:
    HuggingFaceHub.model_rebuild(force=True)
except Exception:
    pass
# ───────────────────────────────────────────────────────────────────────────────

# Hugging Face token & embeddings ---------------------------------------------
HF_TOKEN = st.secrets["API_TOKEN"]           # define in .streamlit/secrets.toml
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)

# Prompt templates -------------------------------------------------------------
PERSONA_TEMPLATES = {
    "Friendly": '''Answer the question in a warm, conversational tone based ONLY on the following context:
    {context}

    Question: {question}

    If you don't know the answer, just say that you don't know. Keep it friendly and approachable!
    ''',
    "Formal": '''Answer the question in a professional and respectful tone based ONLY on the following context:
    {context}

    Question: {question}

    If you don't know the answer, state it politely without making up any information.
    ''',
    "Technical": '''Answer the question in a technical and detailed tone based ONLY on the following context:
    {context}

    Question: {question}

    Provide accurate, in‑depth answers as applicable. Do not guess if the answer is not in the context.
    ''',
    "Concise": '''Answer the question briefly and to the point based ONLY on the following context:
    {context}

    Question: {question}

    Keep responses short and straightforward. Only answer based on the context provided.
    ''',
}

# Session state defaults -------------------------------------------------------
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("temperature_value", 0.5)
st.session_state.setdefault("persona", "Technical")
session_id = st.session_state.setdefault("session_id", str(uuid.uuid4()))

# Sidebar UI -------------------------------------------------------------------
file_upload = st.sidebar.file_uploader("Upload your PDF", type="pdf")

st.session_state["temperature_value"] = st.sidebar.slider(
    "LLM Model Temperature",
    0.05,
    1.0,
    st.session_state["temperature_value"],
    0.05,
)
st.sidebar.write("Lower temperature → answers stick more closely to your PDF content.")

st.session_state["persona"] = st.sidebar.selectbox(
    "Assistant tone",
    ("Friendly", "Formal", "Technical", "Concise"),
    index=("Friendly", "Formal", "Technical", "Concise").index(st.session_state["persona"]),
)

st.sidebar.divider()

if st.sidebar.button("Delete PDF contents from vector DB"):
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]
        st.sidebar.success("Collection deleted.")
    else:
        st.sidebar.error("No collection to delete.")

if st.session_state["chat_history"]:
    history_json = json.dumps(st.session_state["chat_history"], indent=2)
    st.sidebar.download_button(
        "Download Chat History",
        history_json,
        f"History_{datetime.now():%Y%m%d}.json",
        "application/json",
    )

# Vector DB --------------------------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = Chroma(
        embedding_function=embeddings,
        collection_name=session_id,
    )
retriever = st.session_state["vector_db"].as_retriever()

# Initialise HuggingFace LLM ---------------------------------------------------
repo_id = "huggingfaceh4/zephyr-7b-alpha"
model_kwargs = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": st.session_state["temperature_value"],
    "top_p": 0.9,
    "return_full_text": False,
}

try:
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
except Exception as e:
    st.error(f"🚫 Could not initialise Hugging Face model:\n{e}")
    st.stop()

# Build RAG chain --------------------------------------------------------------
prompt_tpl = ChatPromptTemplate.from_template(PERSONA_TEMPLATES[st.session_state["persona"]])
output_parser = StrOutputParser()

chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt_tpl
    | llm
    | output_parser
)

# Main UI ----------------------------------------------------------------------
st.title("Chat with PDF")

# Handle PDF upload ------------------------------------------------------------
if file_upload:
    st.session_state["file_name"] = file_upload.name

    # Refresh vector store
    del st.session_state["vector_db"]
    st.session_state["vector_db"] = Chroma(
        embedding_function=embeddings,
        collection_name=session_id,
    )
    retriever = st.session_state["vector_db"].as_retriever()

    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, f"{session_id}_{file_upload.name}")
    with open(pdf_path, "wb") as f:
        f.write(file_upload.getvalue())

    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128).split_documents(docs)
    st.session_state["vector_db"] = Chroma.from_documents(chunks, embeddings, collection_name=session_id)
    shutil.rmtree(tmp_dir)

# Chat interface ---------------------------------------------------------------
if file_upload:
    msg_container = st.container(height=600, border=True)

    for m in st.session_state["chat_history"]:
        avatar = "🤖" if m["role"] == "assistant" else "🤔"
        with msg_container.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    if user_msg := st.chat_input("Enter a prompt here…"):
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        msg_container.chat_message("user", avatar="🤔").markdown(user_msg)

        with msg_container.chat_message("assistant", avatar="🤖"):
            with st.spinner("processing…"):
                answer = chain.invoke(user_msg) if st.session_state["vector_db"] else "Please upload a PDF first."
                st.markdown(answer)

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()
else:
    st.info("⬅️ Upload a PDF from the sidebar to start chatting.")
