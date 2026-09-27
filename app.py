"""
app.py - The Streamlit chat interface.

Loads the chunks we stored in ingest.py, takes a user question,
finds the most relevant chunks, sends them + the question to DeepSeek,
and displays the answer.

Run with:  streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------------------------------------------------
# SETUP - runs once when the app starts
# ------------------------------------------------------------------
load_dotenv()  # loads DEEPSEEK_API_KEY from .env

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "return_policies"
TOP_K = 3                     # how many chunks to retrieve per question
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEEPSEEK_MODEL = "deepseek-chat"


# ------------------------------------------------------------------
# CACHE RESOURCES - Streamlit reruns the whole script on every
# interaction, so we cache heavy objects (model + DB) to avoid
# reloading them on every question.
# ------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


@st.cache_resource
def load_llm_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY not found in .env file")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ------------------------------------------------------------------
# RETRIEVAL - find the most relevant chunks for a question
# ------------------------------------------------------------------
def retrieve_context(question: str, embedder, collection, k: int = TOP_K):
    """
    Embeds the question using the SAME model we used for chunks.
    
    CRITICAL: both the question and the chunks must live in the same
    vector space, which is why we must use the same embedding model.
    Using a different model would give meaningless similarity scores.
    """
    question_vec = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_vec,
        n_results=k,
        include=["documents", "distances"]
    )

    # results["documents"][0] is the list of chunks for our single query
    docs = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(docs, distances))


# ------------------------------------------------------------------
# GENERATION - send retrieved chunks + question to DeepSeek
# ------------------------------------------------------------------
def ask_deepseek(question: str, context_chunks: list[str], llm_client) -> str:
    """
    The 'augmented prompt' - this is the heart of RAG.
    
    We give the LLM:
    1. A clear instruction: answer ONLY from the context
    2. The retrieved context chunks
    3. The user's question
    
    The instruction is critical: without "only use the context below",
    the LLM may invent return policies from its training data.
    """
    context_block = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful customer support assistant for an e-commerce "
        "return policy. Answer the user's question using ONLY the policy "
        "context provided below. If the answer is not in the context, say "
        "'I couldn't find that in the policy. Please contact support.' "
        "Do not make up information. Be concise and quote specific "
        "numbers (days, refund amounts) when relevant."
    )

    user_prompt = f"""POLICY CONTEXT:
{context_block}

USER QUESTION: {question}

Answer based only on the context above."""

    response = llm_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2   # low = factual, less creative
    )

    return response.choices[0].message.content


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Return Policy Assistant", page_icon="🛒")
    st.title("🛒 Return & Refund Assistant")
    st.caption("Ask anything about the return policy — get answers in plain English.")

    embedder = load_embedder()
    collection = load_collection()
    llm_client = load_llm_client()

    # Session state keeps the chat history across reruns
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Source chunks used"):
                    for i, (doc, dist) in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}** (similarity distance: {dist:.3f})")
                        st.text(doc)

    # Chat input
    if question := st.chat_input("e.g. I got a broken phone. Can I return it?"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Searching policy..."):
                retrieved = retrieve_context(question, embedder, collection)
                context_chunks = [doc for doc, _ in retrieved]
                answer = ask_deepseek(question, context_chunks, llm_client)

            st.markdown(answer)

            with st.expander("📄 Source chunks used"):
                for i, (doc, dist) in enumerate(retrieved, 1):
                    st.markdown(f"**Chunk {i}** (similarity distance: {dist:.3f})")
                    st.text(doc)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved
        })


if __name__ == "__main__":
    main()