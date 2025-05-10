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

# ─── Session‑level objects ─────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "temperature_value" not in st.session_state:
    st.session_state.temperature_value = 0.5

if "persona" not in st.session_state:
    st.session_state.persona = "Technical"

# ─── Sidebar controls ──────────────────────────────────────────────────────────
file_upload = st.sidebar.file_uploader("Upload your PDF", type="pdf")

st.session_state.temperature_value = st.sidebar.slider(
    "LLM Model Temperature",
    min_value=0.05,
    max_value=1.0,
    value=st.session_state.temperature_value,
    step=0.05,
)
st.sidebar.write(
    "Lower temperature → answers stick more closely to your PDF content."
)

st.session_state.persona = st.sidebar.selectbox(
    "Assistant tone",
    ("Friendly", "Formal", "Technical", "Concise"),
    index=("Friendly", "Formal", "Technical", "Concise").index(
        st.session_state.persona
    ),
)

st.sidebar.divider()

# Button to wipe vector DB (PDF contents)
if st.sidebar.button("Delete PDF contents from vector DB"):
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]
        st.sidebar.success("Collection deleted.")
    else:
        st.sidebar.error("No collection to delete.")

# Button to download chat history
if st.session_state.chat_history:
    history_json = json.dumps(st.session_state.chat_history, indent=2)
    st.sidebar.download_button(
        label="Download Chat History",
        data=history_json,
        file_name=f"History_{datetime.now():%Y%m%d}.json",
        mime="application/json",
    )

# ─── Unique session ID & Chroma collection ────────────────────────────────────
session_id = st.session_state.get("session_id") or str(uuid.uuid4())
st.session_state.session_id = session_id

if "vector_db" not in st.session_state:
    st.session_state.vector_db = Chroma(
        embedding_function=embeddings,
        collection_name=session_id,
    )
retriever = st.session_state.vector_db.as_retriever()

# ─── Initialise LLM FIRST (before building chain) ─────────────────────────────
repo_id = "huggingfaceh4/zephyr-7b-alpha"
model_kwargs = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": st.session_state.temperature_value,
    "top_p": 0.9,
    "return_full_text": False,
}

try:
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)
except Exception as e:
    st.error(f"🚫 Unable to initialise Hugging Face model: {e}")
    st.stop()                           # Prevents NameError further down

# ─── Build prompt, parser, and chain ──────────────────────────────────────────
prompt_tpl = ChatPromptTemplate.from_template(
    PERSONA_TEMPLATES[st.session_state.persona]
)
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

# ─── Main title ───────────────────────────────────────────────────────────────
st.title("Chat with PDF")

# ─── Handle PDF upload ────────────────────────────────────────────────────────
if file_upload:
    # Keep original name for history download
    st.session_state.file_name = file_upload.name

    # Reset collection on every new upload
    del st.session_state.vector_db
    st.session_state.vector_db = Chroma(
        embedding_function=embeddings, collection_name=session_id
    )
    retriever = st.session_state.vector_db.as_retriever()

    # Persist file to temp dir (needed for PyPDFLoader)
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, f"{session_id}_{file_upload.name}")
    with open(pdf_path, "wb") as f:
        f.write(file_upload.getvalue())

    # Load, split, embed, store
    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=128
    )
    chunks = splitter.split_documents(docs)
    st.session_state.vector_db = Chroma.from_documents(
        chunks, embeddings, collection_name=session_id
    )
    shutil.rmtree(tmp_dir)

# ─── Chat interface ───────────────────────────────────────────────────────────
if file_upload:
    msg_container = st.container(height=600, border=True)

    # Display history
    for m in st.session_state.chat_history:
        avatar = "🤖" if m["role"] == "assistant" else "🤔"
        with msg_container.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # New prompt
    if user_msg := st.chat_input("Enter a prompt here…"):
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg}
        )
        msg_container.chat_message("user", avatar="🤔").markdown(user_msg)

        with msg_container.chat_message("assistant", avatar="🤖"):
            with st.spinner("processing…"):
                if st.session_state.vector_db is not None:
                    answer = chain.invoke(user_msg)
                    st.markdown(answer)
                else:
                    st.warning("Please upload a PDF first.")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()
else:
    st.info("⬅️ Upload a PDF from the sidebar to start chatting.")
