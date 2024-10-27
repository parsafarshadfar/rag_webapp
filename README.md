
# RAG Webapp

This repository contains a Streamlit web application that serves as a retrieval-augmented generation (RAG) tool for PDF documents. The application leverages a language model to enable efficient querying of PDF content, providing a responsive and private environment for document analysis.

## Features
- **Local Document Analysis**: Upload PDF files to get extracted and analyzed content locally. Users can also drag and drop PDF files
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with generation for relevant, context-aware responses.
- **Interactive Chat Interface**: Once a PDF is uploaded, users can input questions or prompts related to the document’s content, and the app provides responses based on the embedded document data. The app’s interface is designed for simplicity and functionality!
- **temperature control** : 
- **Tone control**:
- **Download chat history**: download the chat history. it is in openAI syntax format.


## Setup

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit application by running:
```bash
streamlit run webapp.py

or 

python3 -m streamlit run webapp.py
```

This will launch the application on `http://localhost:8501`, where you can upload PDF files and query content locally.


![An overview of the the RAG Webapp: upload PDF, Ask questions.](./Screenshot.png)

## Usage

1. **Upload PDFs**: Drag and drop PDF files directly into the web application.
2. **Query the Document**: Enter a query in the text box, and the application will retrieve and generate responses based on the uploaded document's content.
3. **Download Results**: Optionally download generated responses or processed text for further use.

## Acknowledgments