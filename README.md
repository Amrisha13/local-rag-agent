# Local RAG Chat - Restaurant Review Q&A System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a pizza restaurant using local LLMs via Ollama. The system uses vector embeddings to retrieve relevant restaurant reviews and generates contextual answers.

## Features

- **Local LLM Integration**: Uses Ollama with Llama 3.2 for response generation
- **Vector Database**: ChromaDB for efficient similarity search
- **RAG Architecture**: Combines retrieval and generation for accurate, context-aware answers
- **Restaurant Review Analysis**: Processes and queries restaurant reviews with ratings and dates
- **Interactive CLI**: Simple command-line interface for asking questions

## Architecture

The project consists of two main components:

1. **Vector Store (`vector.py`)**: 
   - Loads restaurant reviews from CSV
   - Creates embeddings using Ollama's `mxbai-embed-large` model
   - Stores vectors in ChromaDB for fast retrieval
   - Provides a retriever interface for similarity search

2. **Chat Interface (`main.py`)**:
   - Interactive Q&A loop
   - Retrieves relevant reviews based on user questions
   - Generates responses using Llama 3.2 LLM
   - Combines retrieved context with user queries

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

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your restaurant reviews CSV file is in the project directory:
   - File: `realistic_restaurant_reviews.csv`
   - Required columns: `Title`, `Review`, `Rating`, `Date`

## Usage

1. **First Run** (Vector Store Creation):
   - On first run, the system will create the vector database
   - Reviews will be embedded and stored in `./chroma_langchain_db/`
   - This process may take a few minutes depending on dataset size

2. **Start the Chat Interface**:
```bash
python main.py
```

3. **Ask Questions**:
```
Question: What do customers say about the pizza quality?
Question: Are there any complaints about service?
Question: What are the most common positive reviews?
```

4. **Exit**: Type `q` to quit the application

## Project Structure

```
local-rag-chat/
├── main.py                          # Main chat interface
├── vector.py                        # Vector store setup and retriever
├── requirements.txt                 # Python dependencies
├── realistic_restaurant_reviews.csv # Restaurant reviews dataset
├── chroma_langchain_db/            # ChromaDB vector store (auto-generated)
└── README.md                        # This file
```

## Configuration

### Modify LLM Model
Edit `main.py` to change the language model:
```python
model = OllamaLLM(model="llama3.2")  # Change to your preferred model
```

### Adjust Retrieval Parameters
Edit `vector.py` to change the number of retrieved reviews:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Change k value
```

### Customize Prompt Template
Edit the template in `main.py`:
```python
template = """
You are an expert in answering questions about a pizza restaurant.
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""
```

## Data Format

The CSV file should contain the following columns:

- **Title**: Review title/summary
- **Review**: Full review text
- **Rating**: Numerical rating (e.g., 1-5 stars)
- **Date**: Review date

## Technologies Used

- **LangChain**: Framework for LLM applications
- **Ollama**: Local LLM runtime
- **ChromaDB**: Vector database for embeddings
- **Pandas**: Data processing
- **Python**: Core programming language

## Future Enhancements

- [ ] Web interface using Streamlit or Gradio
- [ ] Support for multiple data sources
- [ ] Conversation history and context
- [ ] Advanced filtering by rating/date
- [ ] Export conversation logs
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This project runs entirely locally and does not send data to external APIs, ensuring privacy and data security.