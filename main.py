from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, prompt
from vector import retriever

model = OllamaLLM(model= "llama3.2")
template = """
You are a helpful assistant analyzing restaurant reviews for a pizza restaurant.

Context - Relevant Customer Reviews:
{reviews}

User Question: {question}

Instructions:
- Base your answer primarily on the provided reviews
- If the reviews don't contain relevant information, acknowledge this
- Summarize key points from multiple reviews when applicable
- Mention specific details like ratings or dates if relevant
- Be concise and factual

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
    print("\n\n ----------------")
    question = input("Question: ")
    if question == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews":[reviews], "question": question})
    print(result)