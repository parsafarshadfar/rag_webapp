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

from langchain_community.llms import Cohere
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma



#setting streamlit main page configration
st.set_page_config(
    page_title="RAG Webapp",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)



############################ LLM Model ###########################

# hugging face model configration info:
HF_TOKEN = st.secrets['API_TOKEN'] # add you hugging face token here, this line should be sth like this: HF_TOKEN ='hfra......sbhfasjfhJ'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)

print("Retrieving...")


session_id = str(uuid.uuid4())  # a unique 36-char string for each online user # to have a separate database in chroma 

#initializing a chroma vector database in the streamlit cached memory
st.session_state["vector_db"] = Chroma( embedding_function=embeddings, collection_name= session_id )
#define the retriever function
retriever = st.session_state["vector_db"].as_retriever()



# template = """ Answer the question based ONLY on the following context:
#     {context}

#     Question: {question}
    
#     If you don't know the answer, just say that you don't know, don't try to make up an answer.
#     Only provide the answer from the {context}, nothing else.
#     Add snippets of the context you used to answer the question.

# """



# Define persona-specific prompt templates
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
    
    Provide accurate, in-depth answers as applicable. Do not guess if the answer is not in the context.
    """,
    
    "Concise": """Answer the question briefly and to the point based ONLY on the following context:
    {context}

    Question: {question}
    
    Keep responses short and straightforward. Only answer based on the context provided.
    """
}



# Model Configration
llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={
        "max_new_tokens":512, #max response length
        "repetition_penalty": 1.1, #parameter is used to discourage the language model from repeating the same words, phrases, or sentences in its responses
        "temperature": st.session_state.get('temperature_value',0.5), # to force the model to only answrer based on the pdf file, you can reduce the temperature
        "top_p": 0.9, #  letting the model consider a wider range of words (top_p : 0 to 1)
        "return_full_text":False}
)

# prompt = ChatPromptTemplate.from_template(template) # make the prompt template based on 'context' and 'question'
prompt = ChatPromptTemplate.from_template(persona_templates[st.session_state.get('persona','Technical')])  # make the persona specific prompt template based on 'context' and 'question'
output_parser = StrOutputParser() #output


#The main RAG pipeline from input to output to be called in future
chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | output_parser
)


################################# WebApp ###################################
st.title("Chat with PDF") # main page heading


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


#upload section in the sidebar
file_upload = st.sidebar.file_uploader("Upload your PDF", type='pdf' )
if file_upload:

    #save the file name:
    st.session_state['file_name'] = file_upload.name

    #first empty the database at each upload
    if 'vector_db' in st.session_state:
        del st.session_state["vector_db"]

    # then make a refresh database with chroma
    st.session_state["vector_db"] = Chroma( embedding_function=embeddings, collection_name= session_id )
    retriever = st.session_state["vector_db"].as_retriever()

    #save the file in an actual path (but temporary) before feeding to PyPDFLoader
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, session_id + '_' + file_upload.name)

    with open(path, "wb") as w:
        w.write(file_upload.getvalue())
    
    #load the pdf
    print("Loading data ..")
    data = PyPDFLoader(path)
    content = data.load()

    #chunk the data
    print("Splitting data...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2048,chunk_overlap=128)
    chunks = splitter.split_documents(content)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
    )

    #add the embeddings and chunks to the vector db
    print("Save to vector DB..")
    st.session_state["vector_db"]  = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=session_id
    )

    print("vector DB Created..")

    shutil.rmtree(temp_dir) #remove the temporary directory created for the file parsing


st.sidebar.divider() # add a divider in the sidebar panel
# Set up a temperature slider for the LLM model
st.session_state['temperature_value'] = st.sidebar.slider(
    "LLM Model Temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,  # Default value
    step=0.05
)
st.sidebar.write('Note: Lower the temperature for responses that adhere strictly to your PDF content.')

st.sidebar.divider() # add a divider in the sidebar panel


st.session_state['persona'] = st.sidebar.selectbox(
    "Assistant's tone:",
    ("Friendly", "Formal", "Technical", "Concise"),
    index=2  # Set default to "Concise" (0-based index, so "Technical" is 2)
)

st.sidebar.divider() # add a divider in the sidebar panel


# Delete PDF contents from vector DB if possible
delete_collection = st.sidebar.button("Delete PDF contents from vector DB")
if delete_collection:
    if 'vector_db' in st.session_state:
        del st.session_state["vector_db"]
        st.sidebar.success("Collection deleted successfully.")
    else:
        st.sidebar.error("No vector database found to delete.")


# Download button to download chat history as JSON in OpenAI format
if st.session_state.chat_history: # if there is a chat history:
    
    chat_history_json = json.dumps(st.session_state.chat_history, indent=4) # create the json file
    
    if st.sidebar.download_button(
        label="Download Chat History",
        data=chat_history_json,
        file_name= "History_" + st.session_state.get('file_name','') +'_'+ datetime.now().strftime("%Y%m%d") +".json", 
        mime="application/json"
    ):
        # Show balloons as soon as the download button is clicked
        st.balloons()



### main chatting area 
if file_upload:

    ## simple chat:
    # prompt = st.text_input("Enter your prompt:")
    # text_container = st.empty()
    # if prompt:
    #     response = chain.invoke(prompt) # call the chain pipeline and get the response from the model
    #     st.write(response) # Print model's response in the webapp


    ## Fancy chat box with history and scrolling:
    message_container = st.container(height=600, border=True)

    for message in st.session_state["chat_history"]:
        avatar = "🤖" if message["role"] == "assistant" else "🤔"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        try:
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="🤔").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    if st.session_state["vector_db"] is not None:
                        response = chain.invoke(prompt) 
                        st.markdown(response)
                    else:
                        st.warning("Please upload a PDF file first.")

            if st.session_state["vector_db"] is not None:
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()

        except Exception as e:
            st.error(e)







