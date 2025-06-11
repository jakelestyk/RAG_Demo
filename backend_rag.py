import os
import re
import fitz
import chromadb
import requests
import json
import argparse
from sentence_transformers import SentenceTransformer
import torch
import logging
from typing import Tuple, List

# ---- Configuration and Constraints ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PDF_DIRECTORY = "docs"
DB_PATH = "chroma_db"
COLLECTION_NAME = "example_health_docs"

EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

OLLAMA_MODEL_NAME = "gemma3:latest"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

CHUNK_SIZE = 768
CHUNK_OVERLAP = 75


# ---- Text extraction and cleaning ----
def clean_pdf_text(text):
    text = re.sub(r'MAYO\s*CLINIC.*(?:\n.*)*?(?:Request an Appointment|Log in|Symptoms &|causes|Diagnosis &|treatment|Doctors &|departments)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2,2} [AP]M', '', text)
    text = re.sub(r'\d+/\d+', '', text)
    
    text = re.sub(r'Request an appointment', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Print', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Show references', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Advertisement', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Close', '', text, flags=re.IGNORECASE)
    text = re.sub(r'By Mayo Clinic Staff', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Enlarge image', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Image \d+\]', '', text)
    
    text = re.sub(r'From Mayo Clinic to your inbox.*Subscribe!', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not re.match(r'^(Overview|Symptoms|Causes|Risk factors|Complications|Prevention|Diagnosis|Treatment|Doctors & departments|When to see a doctor)\s*↓*$', line.strip())]
    text = '\n'.join(cleaned_lines)
    return text.strip()

def load_and_process_pdfs(directory):
    documents = []
    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist.")
        return documents

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(directory, filename)
            logging.info(f"Processing PDF: {filename}")
            
            try:
                doc = fitz.open(path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                cleaned_text = clean_pdf_text(full_text)
                for i in range(0, len(cleaned_text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = cleaned_text[i:i + CHUNK_SIZE]
                    documents.append({
                        "content":chunk,
                        "metadata":{
                            "source":filename
                        }
                    })
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    logging.info(f"Successfully processed {len(os.listdir(directory))} PDFs into {len(documents)} chunks.")
    return documents


#---- Vector database setup ----
def setup_chromadb(documents, embedding_model, rebuild=False):
    client = chromadb.PersistentClient(path=DB_PATH)
    
    if rebuild:
        try:
            if COLLECTION_NAME in [c.name for c in client.list_collections()]:
                logging.info(f'Rebuilding DB... Delecting existing collection {COLLECTION_NAME}')
                client.delete_collection(name=COLLECTION_NAME)
        except Exception as e:
            logging.error(f"Error in deleting collection: {e}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    if collection.count() == 0 or rebuild:
        if not documents:
            logging.warning(f"There are 0 documents to populate the database with. Please check you have added documents to the {PDF_DIRECTORY} directory.")
            return collection
        logging.info(f"Database is empty or rebuild is forced. Populating with new data...")
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            embeddings = embedding_model.encode(batch, convert_to_tensor=True).tolist()
            all_embeddings.extend(embeddings)
            logging.info(f"Embedded batch {i//batch_size+1}/{(len(contents) + batch_size - 1)//batch_size}")
        
        collection.add(
            embeddings=all_embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(documents)} chunks to ChromaDB")
    else:
        logging.info("Existing documents found and loaded")
    
    return collection


#---- Retrieval and Generation ----
def retrieve_context(query, collection, embedding_model, n_results=5):
    logging.info(f"Retrieving context for query: '{query}'")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    context = "\n\n---\n\n".join(results['documents'][0])
    sources = sorted(list(set(meta['source'] for meta in results['metadatas'][0])))
    logging.info(f"Retrieved context from sources: {list(sources)}")
    
    return context, sources

def generate_answer(query, context, sources):
    sources_text = "\n".join(f"- {source}" for source in sources)
    prompt_template = f"""
    You are a helpful medical information assistant. 
    Your task is to answer the user's question based *only* on the provided context from Mayo Clinic documents. 
    After providing the answer, you MUST list the sources you used under a 'Sources:' heading. Use the list of filenames provided below. 
    If the information is not in the context, explicitly state that you cannot answer the question with the given information and do not list any sources. 
    
    CONTEXT: {context}
    
    SOURCES: {sources_text}
    
    QUESTION: {query}
    
    ANSWER:
    """
    logging.info("Sending prompt to Ollama LLM")
    try:
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt_template,
            "stream": False   # For a single response object
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()  # Check if the request was successful or raise an exception
        
        response_data = response.json()
        return response_data.get('response', "Error: Could not extract response from Ollama model").strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return "Error: Could not connect to the Ollama server. Please ensure Ollama is running."
    except json.JSONDecodeError:
        logging.error(f"Failed to decode Ollama's response: {response.text}")
        return "Error: Received an invalid response from Ollama"


#---- Main Execution Phase ----

def main():
    parser = argparse.ArgumentParser(description="A RAG chatbot for patient and family healthcare education")
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Force the reprocessing of PDFs and rebuilding of the vector database."
    )
    args = parser.parse_args()
    
    try:
        requests.get("http://localhost:11434")
    except requests.exceptions.ConnectionError:
        logging.error("Ollama server is not found. Please start Ollama and pull a model (e.g. 'ollama run llama-4-scout'.)")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on device '{device}'")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    documents=[]
    db_exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0
    if args.rebuild_db or not db_exists:
        documents = load_and_process_pdfs(PDF_DIRECTORY)
        if not documents:
            logging.error(f"No documents were found or processed from the {PDF_DIRECTORY} directory. Please check the directory, add your PDFs, and try again.")
            return
    
    collection = setup_chromadb(documents, embedding_model, rebuild=args.rebuild_db)
    print(f"DEBUG: Found {collection.count()} documents in the ChromaDB collection.\n")
    
    print("\n--- Patient Health Education RAG Chatbot Demonstration ---")
    print(f"LLM: {OLLAMA_MODEL_NAME} | Embedding: PubMedBERT | DB: ChromaDB")
    print("Type exit to quit this application.")
    
    while True:
        try:
            query = input("\nPlease ask a medical education-related question! \nWe have information on Allergies, Ear infections, Type 1 diabetes, Pneumonia, Influenza, the Common Cold, and many more.\n\n")
            if query.lower() == "exit":
                break
            if not query.strip():
                continue
            
            context, sources = retrieve_context(query, collection, embedding_model)
            answer = generate_answer(query, context, sources)
            
            print("\nAnswer:")
            print(answer)
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            
if __name__ == "__main__":
    main()