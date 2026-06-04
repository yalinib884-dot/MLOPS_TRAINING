import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env variables
load_dotenv()

INDEX_NAME = "healthcareindex"

# Page Config
st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

# Header
st.title("🩺 Healthcare RAG Assistant")
st.markdown(
    "Ask healthcare and medicine-related questions based on uploaded medical documents."
)

# Sidebar
with st.sidebar:
    st.header("Healthcare RAG")
    st.write(
        """
        This application uses:

        - Gemini 2.5 Flash
        - Pinecone Vector Database
        - MiniLM Embeddings
        - Medical PDF Documents
        """
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Load models once
@st.cache_resource
def load_components():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    return vectorstore, llm


vectorstore, llm = load_components()

# User Input
if question := st.chat_input("Ask a healthcare question..."):

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Greeting Handling
    greetings = [
        "hi",
        "hii",
        "hiii",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if question.lower().strip() in greetings:

        answer = (
            "Hello! 👋\n\n"
            "I am your Healthcare RAG Assistant.\n\n"
            "Ask me any healthcare or medicine-related question."
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    else:

        with st.chat_message("assistant"):

            with st.spinner("Searching medical documents..."):

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

1. First use the provided medical document context.

2. If the answer exists in the context:
   - Answer using the context.

3. If the answer is not available in the context:
   - Mention that it was not found in the uploaded medical documents.
   - Then answer using your general medical knowledge.

4. If the user asks a non-medical question:
   - Politely explain that you only answer healthcare-related questions.

5. Keep answers concise, accurate, and professional.

Medical Document Context:
{context}

Question:
{question}

Answer:
"""

                    response = llm.invoke(prompt)

                    answer = response.content

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    error_message = f"Error: {str(e)}"

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )