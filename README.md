# Local RAG Chat - PDF Question & Answer System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about PDF documents using local LLMs via Ollama. The system uses vector embeddings to retrieve relevant content from PDFs and generates contextual answers.

## Features

- **PDF Document Processing**: Automatically loads and processes PDF files
- **Smart Caching**: Only reloads PDFs when they change, improving performance
- **Text Chunking**: Intelligently splits large documents into manageable chunks
- **Local LLM Integration**: Uses Ollama with Llama 3.2 for response generation
- **Vector Database**: ChromaDB for efficient similarity search
- **RAG Architecture**: Combines retrieval and generation for accurate, context-aware answers
- **Interactive CLI**: Simple command-line interface for asking questions

## Architecture

The project consists of two main components:

1. **Vector Store (`vector.py`)**: 
   - Loads PDF documents and splits them into chunks
   - Creates embeddings using Ollama's `mxbai-embed-large` model
   - Stores vectors in ChromaDB for fast retrieval
   - Tracks PDF modification time to avoid unnecessary reloading
   - Provides a retriever interface for similarity search

2. **Chat Interface (`main.py`)**:
   - Interactive Q&A loop
   - Retrieves relevant document chunks based on user questions
   - Generates responses using Llama 3.2 LLM
   - Shows source and page numbers for transparency

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models:
  - `llama3.2` (for text generation)
  - `mxbai-embed-large` (for embeddings)

### Install Ollama Models

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd local-rag-chat
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Place your PDF file in the project directory:
   - Default filename: `sample.pdf`
   - You can change this in `vector.py`

## Usage

1. **First Run** (Vector Store Creation):
   - On first run, the system will process your PDF
   - Text will be extracted, chunked, embedded, and stored in `./chroma_langchain_db/`
   - PDF metadata is saved in `pdf_metadata.json` to track changes
   - This process may take a few minutes depending on PDF size

2. **Start the Chat Interface**:
```bash
python main.py
```

3. **Ask Questions**:
```
Question: What is the main topic of this document?
Question: Can you summarize the key points?
Question: What does it say about [specific topic]?
```

4. **Exit**: Type `q` to quit the application

## Smart Caching

The system automatically detects when your PDF has changed:
- **Same PDF**: Uses cached embeddings (instant startup)
- **Modified PDF**: Automatically reloads and re-embeds the document
- **New PDF**: Replace `sample.pdf` and the system will detect the change

## Project Structure

```
local-rag-chat/
├── main.py                    # Main chat interface
├── vector.py                  # Vector store setup and retriever
├── requirements.txt           # Python dependencies
├── sample.pdf                 # Your PDF document
├── pdf_metadata.json          # PDF change tracking (auto-generated)
├── chroma_langchain_db/       # ChromaDB vector store (auto-generated)
└── README.md                  # This file
```

## Configuration

### Change PDF File
Edit `vector.py` to use a different PDF:
```python
pdf_path = "your_document.pdf"  # Change filename here
```

### Modify LLM Model
Edit `main.py` to change the language model:
```python
model = OllamaLLM(model="llama3.2")  # Change to your preferred model
```

### Adjust Chunking Parameters
Edit `vector.py` to change text splitting:
```python
def load_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    # Adjust chunk_size and chunk_overlap as needed
```

### Adjust Retrieval Parameters
Edit `vector.py` to change the number of retrieved chunks:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Change k value
```

### Customize Prompt Template
Edit the template in `main.py`:
```python
template = """
You are a helpful assistant that answers questions based on provided documents.
Context: {context}
Question: {question}
"""
```

## Technologies Used

- **LangChain**: Framework for LLM applications
- **Ollama**: Local LLM runtime
- **ChromaDB**: Vector database for embeddings
- **PyPDF**: PDF text extraction
- **Python**: Core programming language

## Troubleshooting

### "Input length exceeds context length" Error
- Your PDF pages are too large
- The system automatically chunks text to prevent this
- If you still see this error, reduce `chunk_size` in `vector.py`

### Slow First Run
- First run processes the entire PDF and creates embeddings
- Subsequent runs are much faster due to caching
- Large PDFs (100+ pages) may take several minutes

### PDF Not Reloading
- Delete `pdf_metadata.json` to force a reload
- Or delete the entire `chroma_langchain_db/` directory

## Future Enhancements

- [ ] Support for multiple PDF files
- [ ] Web interface using Streamlit or Gradio
- [ ] Conversation history and context
- [ ] Export conversation logs
- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Advanced filtering by page numbers
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This project runs entirely locally and does not send data to external APIs, ensuring privacy and data security.