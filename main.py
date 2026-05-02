from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from vector import retriever
import sys

model = OllamaLLM(model="llama3.2")

# Template with conversation history
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
chain = prompt | model

# Initialize conversation memory
message_history = ChatMessageHistory()

def format_chat_history():
    """Format chat history for the prompt"""
    if not message_history.messages:
        return "No previous conversation."
    
    history_text = []
    for msg in message_history.messages[-6:]:  # Keep last 3 Q&A pairs (6 messages)
        role = "User" if msg.type == "human" else "Assistant"
        history_text.append(f"{role}: {msg.content}")
    
    return "\n".join(history_text)

print("PDF RAG Chat System with Conversation Memory")
print("Type 'q' to quit, 'clear' to clear conversation history, 'history' to view chat history")
print("=" * 50)

while True:
    print("\n" + "-" * 50)
    question = input("Question: ")
    
    if question.lower() == "q":
        break
    
    if question.lower() == "clear":
        message_history.clear()
        print("✓ Conversation history cleared!")
        continue
    
    if question.lower() == "history":
        print("\n📜 Conversation History:")
        if not message_history.messages:
            print("No conversation history yet.")
        else:
            for i, msg in enumerate(message_history.messages):
                role = "You" if msg.type == "human" else "Assistant"
                print(f"\n{role}: {msg.content}")
        continue
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Format context from retrieved documents
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    # Format chat history
    chat_history = format_chat_history()
    
    # Stream answer from LLM
    print("\nAnswer: ", end="", flush=True)
    
    full_response = ""
    for chunk in chain.stream({
        "context": context,
        "question": question,
        "chat_history": chat_history
    }):
        print(chunk, end="", flush=True)
        full_response += chunk
        sys.stdout.flush()
    
    print()  # New line after streaming completes
    
    # Add to conversation history
    message_history.add_user_message(question)
    message_history.add_ai_message(full_response)
    
    print(f"\n💬 Conversation turns: {len(message_history.messages) // 2}")