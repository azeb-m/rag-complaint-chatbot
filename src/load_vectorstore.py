import os
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings



# Custom embedding class that returns stored embeddings (not used for recomputation)
class PreComputedEmbeddings(Embeddings):
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_documents(self, texts):
        raise NotImplementedError("Precomputed embeddings should be loaded, not recomputed.")

    def embed_query(self, text):
        raise NotImplementedError("Precomputed embeddings should be loaded, not recomputed.")


def build_or_load_vectorstore(parquet_path="data/complaint_embeddings.parquet",
                              persist_directory="vector_store/chroma_db"):
    # Load parquet
    df = pd.read_parquet(parquet_path)

    # Check expected fields
    assert "embedding" in df.columns, "Column 'embedding' not found"
    assert "document" in df.columns, "Column 'document' not found"

    # Choose the text field
    text_field = "document"

    # Get metadata columns (everything except embeddings & text)
    metadata_cols = [
        col for col in df.columns
        if col not in [text_field, "embedding"]
    ]

    documents = []
    metadatas = []
    embeddings = []
    ids = []

    for idx, row in df.iterrows():
        documents.append(row[text_field])
        metadatas.append({k: row[k] for k in metadata_cols})
        embeddings.append(np.array(row["embedding"]).tolist())  # convert to list for Chroma
        ids.append(str(row["id"]) if "id" in df.columns else str(idx))

    # Initialize Chroma client
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory,
    ))

    # Create or get collection
    collection = client.get_or_create_collection(name="complaints")

    # Add documents + embeddings
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # Wrap with LangChain Chroma interface
    vectordb = Chroma(
        client=client,
        collection_name="complaints",
        persist_directory=persist_directory,
        embedding_function=PreComputedEmbeddings(embeddings)
    )

    # Persist to disk
    client.persist()

    return vectordb
