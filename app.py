import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import sys

# Page configuration
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = OllamaLLM(model="llama3.2")

# Template
template = """
You are a helpful assistant that answers questions based on provided documents.

Conversation History:
{chat_history}

Context from Documents:
{context}

User Question: {question}

Instructions:
- Answer based primarily on the provided context
- Use conversation history to understand follow-up questions and maintain context
- If the context doesn't contain relevant information, acknowledge this
- Be concise and factual
- Cite page numbers or sources when available in the metadata
- For follow-up questions, refer back to previous answers when relevant

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | st.session_state.model

def format_chat_history():
    """Format chat history for the prompt"""
    if not st.session_state.messages:
        return "No previous conversation."
    
    history_text = []
    # Keep last 3 Q&A pairs (6 messages)
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text.append(f"{role}: {msg['content']}")
    
    return "\n".join(history_text)

# UI Layout
st.title("📚 PDF RAG Chat")
st.markdown("Ask questions about your PDF documents")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot answers questions based on your PDF documents using:
    - **Llama 3.2** for responses
    - **Vector search** for relevant context
    - **Conversation memory** for follow-ups
    """)
    
    st.divider()
    
    # Conversation stats
    if st.session_state.messages:
        st.metric("Conversation Turns", len(st.session_state.messages) // 2)
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ask specific questions
    - Use follow-up questions
    - Check source citations
    """)

# Main chat area
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 View Sources"):
                    for source in message["sources"]:
                        st.caption(f"**{source['source']}** - Page {source['page']}")
                        st.text(source["content"][:200] + "...")

# Chat input
if question := st.chat_input("Ask a question about your PDF..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Retrieve relevant documents
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(input=question)
        
        # Format context
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # Format chat history
        chat_history = format_chat_history()
        
        # Stream response
        full_response = ""
        for chunk in chain.stream({
            "context": context,
            "question": question,
            "chat_history": chat_history
        }):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # Prepare sources
        sources = [
            {
                "source": doc.metadata.get('source', 'unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "content": doc.page_content
            }
            for doc in docs
        ]
        
        # Show sources
        with st.expander("📄 View Sources"):
            for source in sources:
                st.caption(f"**{source['source']}** - Page {source['page']}")
                st.text(source["content"][:200] + "...")
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })

# Footer
st.divider()
st.caption("🔒 Running locally with Ollama - Your data stays private")
