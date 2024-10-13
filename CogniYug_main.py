"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

Project: Revolutionizing Information Access and Decision-Making with Large Language Models and Retrieval-Augmented Generation

Team Members:
- Samruddhi Ghanwat: samruddighanwat09@gmail.com
- Atharv Prabhune: atharvprabhune04@gmail.com
- Om Gunjal: omgunjalmtt@gmail.com

Case Study: Developing an AI-Powered Knowledge Retrieval System for HR

INTRODUCTION:
This case study outlines the proposed development of an AI-powered knowledge retrieval
system, leveraging Retrieval-Augmented Generation (RAG) technology, aimed at enhancing
decision-making within the Human Resources (HR) department of XYZ Company, a large
multinational corporation. The goal is to streamline access to information, improve HR processes,
and ultimately enhance the employee experience.

PROBLEM STATEMENT:
At XYZ Company, the Human Resources (HR) department plays a crucial role in supporting our
employees and driving the organization’s success. However, several challenges hinder our HR
operations, affecting both our staff and the employee experience.

1. Information Silos:
   - Important HR documents, such as policies, benefits information, and training materials, are scattered across different systems and departments, making it difficult for HR staff to access the information they need promptly.

2. Slow Decision-Making:
   - The lack of easy access to accurate and consolidated data contributes to slow decision-making, affecting recruitment processes and employee growth opportunities.

3. Employee Confusion:
   - Employees often receive inconsistent information about HR policies and procedures, leading to frustration and eroded trust in the HR department.

PROPOSED SOLUTION:
To address these challenges, XYZ Company plans to implement an Enterprise RAG solution. This
system will integrate advanced Large Language Models (LLMs) with a RAG framework to create
an accessible, user-friendly interface for HR knowledge.

Key Features of the Proposed Solution:
1. Centralized Knowledge Repository:
   - All HR-related documents, policies, and guidelines will be indexed into a single database, ensuring that information is easily retrievable.

2. Natural Language Querying:
   - Employees will be able to ask questions using everyday language, enabling faster and more intuitive access to information.

3. Context-Aware Responses:
   - The LLMs will generate tailored responses based on the context of the questions asked, providing clear and relevant information.

"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Streamlit page configuration
st.set_page_config(
    page_title="AI-Powered HR Knowledge Retrieval System",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# CSS for custom styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f2f2f2;
    }
    .header {
        font-size: 2rem;
        color: #34495E;
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-area, .question-area {
        border: 1px solid #BDC3C7;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .btn-success {
        background-color: #2ecc71;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to extract model names from the models info

@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

# Function to create a vector database from the uploaded PDF

def create_vector_db(file_upload) -> Chroma:
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    
    # Split the extracted content into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    
    # Create embeddings for the text chunks
    embeddings = OllamaEmbeddings(model="mistral", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

# Function to process the user's question and generate an answer
def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = ChatOllama(model=selected_model, temperature=0)
    
    # Prompt to generate alternative questions for better document retrieval
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only provide the answer from the {context}, nothing else.
    Add snippets of the context you used to answer the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

# Function to extract all pages of the uploaded PDF as images

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

# Function to delete the vector database and clear session state

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main() -> None:
    st.markdown("<h1 class='header'>Ollama AI-Powered HR Knowledge Retrieval System 🎈</h1>", unsafe_allow_html=True)

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with col1:
        st.header("Upload PDF")
        st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
        file_upload = st.file_uploader(
            "Upload a PDF document", type=["pdf"], label_visibility="collapsed"
        )
        if st.button("Submit PDF", key="submit_pdf") and file_upload is not None:
            vector_db = create_vector_db(file_upload)
            st.session_state["vector_db"] = vector_db
            st.session_state["pdf_pages"] = extract_all_pages_as_images(file_upload)
            st.success("PDF uploaded and vector DB created.", icon="✅")
            st.balloons()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.header("Ask a Question")
        st.markdown("<div class='question-area'>", unsafe_allow_html=True)
        question = st.text_input("What would you like to know?")
        selected_model = st.selectbox(
            "Choose Model", available_models, index=0
        )
        if st.button("Submit Question", key="submit_question") and question:
            vector_db = st.session_state.get("vector_db")
            if vector_db is None:
                st.warning("Please upload a PDF file first.", icon="⚠️")
            else:
                answer = process_question(question, vector_db, selected_model)
                st.success("Here is your answer:", icon="✅")
                st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Delete Vector DB", key="delete_vector_db"):
        vector_db = st.session_state.get("vector_db")
        delete_vector_db(vector_db)

if __name__ == "__main__":
    main()
