from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
import shutil
from pypdf import PdfReader

def load_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    """Load PDF and split into smaller chunks"""
    reader = PdfReader(file_path)
    documents = []
    
    # Extract text from all pages
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # Only add non-empty pages
            doc = Document(
                page_content=text,
                metadata={"source": file_path, "page": page_num + 1}
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

def pdf_has_changed(pdf_path, metadata_file):
    """Check if PDF has been modified since last load"""
    if not os.path.exists(metadata_file):
        return True
    
    current_mtime = os.path.getmtime(pdf_path)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata.get('mtime') != current_mtime

def save_pdf_metadata(pdf_path, metadata_file):
    """Save PDF modification time"""
    mtime = os.path.getmtime(pdf_path)
    with open(metadata_file, 'w') as f:
        json.dump({'mtime': mtime}, f)

# Initialize
pdf_path = "sample.pdf"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_langchain_db"
metadata_file = "./pdf_metadata.json"

# Check if we need to reload
needs_reload = not os.path.exists(db_location) or pdf_has_changed(pdf_path, metadata_file)

if needs_reload:
    # Remove old database
    if os.path.exists(db_location):
        shutil.rmtree(db_location)
    
    # Load PDF documents
    documents = load_pdf(pdf_path)
    
    # Create vector store
    vector_store = Chroma(
        collection_name="pdf_documents", 
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    vector_store.add_documents(documents=documents)
    
    # Save metadata
    save_pdf_metadata(pdf_path, metadata_file)
else:
    # Use existing database
    vector_store = Chroma(
        collection_name="pdf_documents", 
        persist_directory=db_location,
        embedding_function=embeddings
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
