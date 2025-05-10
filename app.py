__import__('pysqlite3') ## line 1 #comment these three lines if you are using it in local pc and dont want to deploy to cloud
import sys ## line 2 #comment these three lines if you are using it in local pc and dont want to deploy to cloud
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') ## line 3 #comment these three lines if you are using it in local pc and dont want to deploy to cloud

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

import os
import streamlit as st
import time
import tempfile
import shutil
import hashlib
import uuid
import json
from datetime import datetime
import requests   # still needed for PDF download during testing, leave it

from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────────────────────────── Streamlit page config ─────────────────────────
st.set_page_config(
    page_title="RAG Webapp",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# ────────────────────────────── LLM & embeddings ─────────────────────────────
HF_TOKEN = st.secrets['API_TOKEN']            # Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="BAAI/bge-base-en-v1.5",
)

session_id = str(uuid.uuid4())               # isolate each user’s collection
st.session_state["vector_db"] = Chroma(
    embedding_function=embeddings,
    collection_name=session_id,
)
retriever = st.session_state["vector_db"].as_retriever()

# Persona‑specific prompt templates
persona_templates = {
    "Friendly": """Answer the question in a warm, conversational tone based ONLY on the following context:
    {context}

    Question: {question}

    If you don't know the answer, just say that you don't know. Keep it friendly and approachable!
    """,
    "Formal": """Answer the question in a professional and respectful tone based ONLY on the following context:
    {context}

    Question: {question}

    If you don't know the answer, state it politely without making up any information.
    """,
    "Technical": """Answer the question in a technical and detailed tone based ONLY on the following context:
    {context}

    Question: {question}

    Provide accurate, in‑depth answers as applicable. Do not guess if the answer is not in the context.
    """,
    "Concise": """Answer the question briefly and to the point based ONLY on the following context:
    {context}

    Question: {question}

    Keep responses short and straightforward. Only answer based on the context provided.
    """,
}

# ────────────────────────────── Hugging Face LLM ──────────────────────────────
repo_id = "huggingfaceh4/zephyr-7b-alpha"
model_kwargs = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": st.session_state.get("temperature_value", 0.5),
    "top_p": 0.9,
    "return_full_text": False,
}

try:
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
except Exception as e:
    st.error(f"🚫 Failed to initialise LLM: {e}")
    st.stop()

prompt = ChatPromptTemplate.from_template(
    persona_templates[st.session_state.get("persona", "Technical")]
)
output_parser = StrOutputParser()

# Build the RAG chain
chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | output_parser
)

# ────────────────────────────── UI & helpers ─────────────────────────────────
st.title("Chat with PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

file_upload = st.sidebar.file_uploader("Upload your PDF", type="pdf")

# Temperature slider
st.session_state["temperature_value"] = st.sidebar.slider(
    "LLM Model Temperature:",
    0.05,
    1.0,
    0.5,
    0.05,
)
st.sidebar.write(
    "Note: Lower the temperature for responses that adhere strictly to your PDF content."
)

# Persona selector
st.session_state["persona"] = st.sidebar.selectbox(
    "Assistant's tone:",
    ("Friendly", "Formal", "Technical", "Concise"),
    index=2,
)

st.sidebar.divider()

# Delete collection button
if st.sidebar.button("Delete PDF contents from vector DB"):
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]
        st.sidebar.success("Collection deleted successfully.")
    else:
        st.sidebar.error("No vector database found to delete.")

# Download chat history
if st.session_state.get("chat_history"):
    history_json = json.dumps(st.session_state.chat_history, indent=4)
    st.sidebar.download_button(
        "Download Chat History",
        history_json,
        f"History_{st.session_state.get('file_name','')}_{datetime.now():%Y%m%d}.json",
        "application/json",
    )

# ────────────────────────────── PDF handling ─────────────────────────────────
if file_upload:
    st.session_state["file_name"] = file_upload.name

    # Refresh vector DB
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]

    st.session_state["vector_db"] = Chroma(
        embedding_function=embeddings, collection_name=session_id
    )
    retriever = st.session_state["vector_db"].as_retriever()

    # Save PDF to temp file
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, f"{session_id}_{file_upload.name}")

    with open(pdf_path, "wb") as f:
        f.write(file_upload.getvalue())

    # Load and split
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128).split_documents(docs)

    # Re‑embed & store
    st.session_state["vector_db"] = Chroma.from_documents(
        chunks, embeddings, collection_name=session_id
    )
    shutil.rmtree(temp_dir)

# ────────────────────────────── Chat interface ───────────────────────────────
if file_upload:
    msg_container = st.container(height=600, border=True)

    # Display history
    for m in st.session_state["chat_history"]:
        avatar = "🤖" if m["role"] == "assistant" else "🤔"
        with msg_container.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # New input
    if prompt_text := st.chat_input("Enter a prompt here…"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt_text})
        msg_container.chat_message("user", avatar="🤔").markdown(prompt_text)

        with msg_container.chat_message("assistant", avatar="🤖"):
            with st.spinner("processing…"):
                if st.session_state["vector_db"]:
                    answer = chain.invoke(prompt_text)
                    st.markdown(answer)
                else:
                    st.warning("Please upload a PDF first.")

        if st.session_state["vector_db"]:
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
