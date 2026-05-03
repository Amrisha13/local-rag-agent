from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
from pypdf import PdfReader
import tempfile

def load_pdf_from_file(file_path, chunk_size=1000, chunk_overlap=200):
    """Load PDF and split into smaller chunks"""
    reader = PdfReader(file_path)
    documents = []
    
    # Extract text from all pages
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            doc = Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path), "page": page_num + 1}
            )
            documents.append(doc)
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def create_vector_store(documents, collection_name="pdf_documents"):
    """Create a new vector store from documents"""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Create temporary directory for this vector store
    db_location = tempfile.mkdtemp(prefix="chroma_")
    
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    vector_store.add_documents(documents=documents)
    
    return vector_store, db_location

def load_existing_vector_store(db_location, collection_name="pdf_documents"):
    """Load an existing vector store"""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    return vector_store

def cleanup_vector_store(db_location):
    """Clean up vector store directory"""
    if os.path.exists(db_location):
        shutil.rmtree(db_location)
