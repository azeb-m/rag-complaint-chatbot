import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def build_faiss_store(embeddings_pickle: str, output_dir: str):
    print("- Loading embeddings...")
    with open(embeddings_pickle, "rb") as f:
        data = pickle.load(f)

    # Unzip into separate lists
    texts, vectors, metadatas = zip(*data)

    print("- Creating FAISS vectorstore...")
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS index from (text, embedding) pairs
    text_embeddings = list(zip(texts, vectors))
    vectorstore = FAISS.from_embeddings(
        text_embeddings,
        embed_model,
        metadatas=list(metadatas),
        ids=None
    )

    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)

    print(f"Vectorstore saved to {output_dir}")
    print(f"Total vectors indexed: {vectorstore.index.ntotal}")

if __name__ == "__main__":
    build_faiss_store(
        embeddings_pickle="data/processed/chunk_embeddings.pkl",
        output_dir="vector_store/faiss_index"
    )
