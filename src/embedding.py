import pickle
from langchain_huggingface import HuggingFaceEmbeddings

def compute_embeddings(input_pickle: str, output_pickle: str):
    """
    Load text chunks and compute vector embeddings.
    """
    print("- Loading text chunks...")
    with open(input_pickle, "rb") as f:
        chunks = pickle.load(f)

    # Use HuggingFace sentence-transformer model
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    print("- Computing embeddings...")
    embeddings = embed_model.embed_documents(texts)

    # Combine them into a list of (text, vector, metadata) tuples
    vector_data = list(zip(texts, embeddings, metadatas))

    print("- Saving embeddings to disk...")
    with open(output_pickle, "wb") as f:
        pickle.dump(vector_data, f)

    print(f"Embeddings saved to {output_pickle}")

if __name__ == "__main__":
    compute_embeddings(
        input_pickle="data/processed/text_chunks.pkl",
        output_pickle="data/processed/chunk_embeddings.pkl"
    )
