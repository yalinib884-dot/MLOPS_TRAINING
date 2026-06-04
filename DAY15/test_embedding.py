from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

result = embeddings.embed_query(
    "What medicine is used for fever?"
)

print(len(result))