from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

models_to_test = [
    "models/embedding-001",
    "embedding-001",
    "models/text-embedding-004"
]

for model in models_to_test:
    try:
        print(f"\nTesting: {model}")

        emb = GoogleGenerativeAIEmbeddings(
            model=model
        )

        vec = emb.embed_query("fever treatment")

        print("SUCCESS")
        print("Dimension:", len(vec))

    except Exception as e:
        print("FAILED")
        print(type(e).__name__)
        print(str(e)[:300])