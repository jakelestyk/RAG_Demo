import os
import logging
import torch
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import main  # Assumes your CLI script is named main.py and in the same directory

# ---- Configurations ----
EMBEDDING_MODEL_NAME = main.EMBEDDING_MODEL_NAME
DB_PATH = main.DB_PATH
PDF_DIRECTORY = main.PDF_DIRECTORY

# ---- Page Setup ----
st.set_page_config(page_title="CHLA Health Education Chatbot", layout="wide")
st.title("CHLA Health Education RAG Chatbot")
st.markdown("Ask questions about family health education and get cited, document-grounded answers.")

# ---- Sidebar ----
st.sidebar.header("Database Configuration")
rebuild_db = st.sidebar.checkbox("Rebuild vector database", value=False)
if st.sidebar.button("Initialize/Refresh Database"):
    with st.spinner("Building or refreshing database..."):
        # Load embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        # Load & process PDFs as needed
        documents = []
        db_exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0
        if rebuild_db or not db_exists:
            documents = main.load_and_process_pdfs(PDF_DIRECTORY)
        # Setup ChromaDB collection
        st.session_state.collection = main.setup_chromadb(documents, st.session_state.embedding_model, rebuild=rebuild_db)
        st.success("Database is ready!")

# Ensure database is initialized
if 'collection' not in st.session_state:
    with st.spinner("Initializing database..."):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        documents = []
        db_exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0
        if rebuild_db or not db_exists:
            documents = main.load_and_process_pdfs(PDF_DIRECTORY)
        st.session_state.collection = main.setup_chromadb(documents, st.session_state.embedding_model, rebuild=rebuild_db)
    st.success("Database initialized!")

# ---- Query Interface ----
query = st.text_input("Enter your question about family health education:")
if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question before submitting.")
    else:
        collection = st.session_state.collection
        embedding_model = st.session_state.embedding_model
        with st.spinner("Retrieving relevant content..."):
            context, sources = main.retrieve_context(query, collection, embedding_model)
        with st.spinner("Generating answer..."):
            answer = main.generate_answer(query, context, sources)
        # Display
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        for src in sources:
            st.markdown(f"- {src}")
