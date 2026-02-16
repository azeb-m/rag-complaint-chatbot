# src/retriever.py
# src/retriever.py

from src.load_vectorstore import build_or_load_vectorstore
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding function once
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vectorstore(use_cache: bool = True):
    """
    Load or build the vector store.
    If use_cache=True, it will reuse the existing store if available.
    """
    return build_or_load_vectorstore()

def retrieve_documents(query: str, k: int = 5):
    """
    Embed the query using all-MiniLM-L6-v2 and perform similarity search
    against the vector store to retrieve the top-k most relevant text chunks.
    """
    vectordb = get_vectorstore()

    # Embed the query
    query_vector = embedding_function.embed_query(query)

    # Run similarity search by vector
    results = vectordb.similarity_search_by_vector(query_vector, k=k)
    return results
