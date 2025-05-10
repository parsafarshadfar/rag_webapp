# ---------------------------------------------------------------
# RAG Web‑app – Streamlit + LangChain + Hugging Face
# ---------------------------------------------------------------
# cloud‑only shim for sqlite3 → pysqlite3 (leave as‑is locally)
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# -------------- stdlib / third‑party imports -------------------
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

import os, time, tempfile, shutil, uuid, json, requests, hashlib
from datetime import datetime
import streamlit as st
from requests.exceptions import HTTPError
from typing import Union                         # ✨ ensure forward‑ref exists

# ✨ ensure Pydantic can resolve 'BaseCache' before model_rebuild
from langchain_core.caches import BaseCache     # ✨ NEW IMPORT

from langchain_community.llms import Cohere, HuggingFaceHub
HuggingFaceHub.model_rebuild()                  # rebuild with all refs resolved

from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------- Streamlit page settings ----------------------
st.set_page_config(
    page_title="RAG Webapp",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# -------------------------- API keys ---------------------------
HF_TOKEN = st.secrets["API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# -------------------- Embeddings & Vector DB -------------------
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)

session_id = str(uuid.uuid4())
st.session_state["vector_db"] = Chroma(
    embedding_function=embeddings, collection_name=session_id
)
retriever = st.session_state["vector_db"].as_retriever()

# ------------------- Persona prompt templates ------------------
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

# ---------------- Proxy helper utilities ----------------------

def fetch_free_proxies():
    proxy_sources = [
        "https://www.proxy-list.download/api/v1/get?type=https",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
        "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/all/data.txt",
    ]
    proxies = []
    for url in proxy_sources:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                for p in resp.text.strip().split("\n"):
                    proxies.append({"http": f"http://{p}", "https": f"http://{p}"})
        except Exception as e:
            st.error(
                f"Error grabbing proxy list (used for HF rate‑limit fallback): {e}"
            )
    return proxies


def test_proxy(proxy):
    """Lightweight HEAD request to check if a proxy works and is fast."""
    test_url = "https://api-inference.huggingface.co/models"
    try:
        start = time.time()
        r = requests.head(test_url, proxies=proxy, timeout=2)
        return r.status_code == 200 and (time.time() - start) < 2
    except Exception:
        return False


def retry_with_proxies(repo_id, model_kwargs):
    """Return a HuggingFaceHub() instance that works through a free proxy."""
    for proxy in fetch_free_proxies():
        if not test_proxy(proxy):
            continue
        try:
            return HuggingFaceHub(
                repo_id=repo_id, model_kwargs=model_kwargs, proxies=proxy
            )
        except HTTPError as e:
            if (
                e.response is not None
                and e.response.status_code == 429
            ):
                # keep trying other proxies
                continue
            raise
    st.error(
        "⚠️ Hugging Face free API rate‑limit hit and no usable proxy found. "
        "Please try again later."
    )
    st.stop()

# ----------------------- LLM initialisation --------------------

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
        st.warning(
            "HF rate‑limit hit – trying to route through free proxies …"
        )
        llm = retry_with_proxies(repo_id, model_kwargs)
    else:
        st.error(f"LLM initialisation failed: {e}")
        st.stop()
except Exception as e:
    st.error(f"Unexpected error while creating the LLM: {e}")
    st.stop()

if llm is None:
    st.error("Could not create the LLM. Check your HF token or internet access.")
    st.stop()

# ---------------------- Prompt & pipeline ----------------------

prompt = ChatPromptTemplate.from_template(
    persona_templates[st.session_state.get("persona", "Technical")]
)
output_parser = StrOutputParser()

chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | output_parser
)

# -------------------------- UI layout --------------------------
st.title("Chat with PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Sidebar: upload PDF --------------------------------------
file_upload = st.sidebar.file_uploader("Upload your PDF", type="pdf")
if file_upload:
    st.session_state["file_name"] = file_upload.name
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]

    st.session_state["vector_db"] = Chroma(
        embedding_function=embeddings, collection_name=session_id
    )
    retriever = st.session_state["vector_db"].as_retriever()

    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, f"{session_id}_{file_upload.name}")
    with open(path, "wb") as w:
        w.write(file_upload.getvalue())

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    chunks = splitter.split_documents(docs)

    st.session_state["vector_db"] = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=session_id
    )

    shutil.rmtree(temp_dir)

# ---- Sidebar: temperature / persona ---------------------------
st.sidebar.divider()
st.session_state["temperature_value"] = st.sidebar.slider(
    "LLM Model Temperature:", 0.05, 1.0, 0.5, 0.05
)
st.sidebar.write(
    "Lower the temperature for responses that adhere strictly to your PDF content."
)
st.sidebar.divider()

st.session_state["persona"] = st.sidebar.selectbox(
    "Assistant's tone:", ("Friendly", "Formal", "Technical", "Concise"), index=2
)
st.sidebar.divider()

# ---- Sidebar: maintenance buttons -----------------------------
if st.sidebar.button("Delete PDF contents from vector DB"):
    st.session_state.pop("vector_db", None)
    st.sidebar.success("Collection deleted successfully.")

if st.session_state.get("chat_history"):
    chat_json = json.dumps(st.session_state.chat_history, indent=4)
    if st.sidebar.download_button(
        "Download Chat History",
        data=chat_json,
        file_name=f"History_{st.session_state.get('file_name','')}_{datetime.now():%Y%m%d}.json",
        mime="application/json",
    ):
        st.balloons()

# ---- Main chat container --------------------------------------
if file_upload:
    msg_container = st.container(height=600, border=True)
    for m in st.session_state["chat_history"]:
        avatar = "🤖" if m["role"] == "assistant" else "🤔"
        with msg_container.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    if prompt_text := st.chat_input("Enter a prompt here…"):
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt_text}
        )
        msg_container.chat_message("user", avatar="🤔").markdown(prompt_text)

        with msg_container.chat_message("assistant", avatar="🤖"):
            with st.spinner(":green[processing…]"):
                if st.session_state["vector_db"] is not None:
                    response_text = chain.invoke(prompt_text)
                    st.markdown(response_text)
                else:
                    st.warning("Please upload a PDF file first.")

        if st.session_state["vector_db"] is not None:
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response_text}
            )
            st.rerun()
