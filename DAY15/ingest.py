import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "healthcareindex"

# Load PDFs
documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        print(f"Loading: {file}")

        loader = PyPDFLoader(
            os.path.join("data", file)
        )

        documents.extend(loader.load())

print(f"\nPages Loaded: {len(documents)}")

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Chunks Created: {len(chunks)}")

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Upload to Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("\nUpload Completed Successfully")