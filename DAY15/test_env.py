from dotenv import load_dotenv
import os

load_dotenv()

print("Google Key:", os.getenv("GOOGLE_API_KEY")[:10])
print("Pinecone Key:", os.getenv("PINECONE_API_KEY")[:10])
