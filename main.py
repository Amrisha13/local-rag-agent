from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a helpful assistant that answers questions based on provided documents.

Context from Documents:
{context}

User Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain relevant information, acknowledge this
- Be concise and factual
- Cite page numbers or sources when available in the metadata

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("PDF RAG Chat System")
print("Type 'q' to quit")
print("=" * 50)

while True:
    print("\n" + "-" * 50)
    question = input("Question: ")
    
    if question.lower() == "q":
        break
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Format context from retrieved documents
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    # Get answer from LLM
    result = chain.invoke({"context": context, "question": question})
    print("\nAnswer:", result)