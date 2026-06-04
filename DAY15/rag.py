import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

INDEX_NAME = "healthcareindex"

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Pinecone Vector Store
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    try:
        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(
            question,
            k=8
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a healthcare assistant.

Rules:

1. First check the provided medical document context.
2. If the answer exists in the context, answer using the context.
3. If the answer is not available in the context:
   - Say that the information was not found in the uploaded medical documents.
   - Then answer using your general medical knowledge.
4. If the user asks a NON-MEDICAL question:
   - Politely say that you are a healthcare assistant and only answer healthcare-related questions.
5. Keep answers concise and accurate.

Medical Document Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        print("\nAnswer:")
        print(response.content)

    except Exception as e:
        print(f"\nError: {e}")