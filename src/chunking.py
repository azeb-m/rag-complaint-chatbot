import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pickle



def chunk_texts(input_csv: str, output_pickle: str):
    df = pd.read_csv(input_csv)
    print(df.columns)

    print("- Generating chunks for each complaint narrative")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # ~500 characters per chunk
        chunk_overlap=50  # ~50 characters overlap
    )

    all_chunks = []
    for idx, row in df.iterrows():
        text = str(row["Consumer complaint narrative"])
        chunks = splitter.create_documents([text])
        for chunk_i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_id": row["Complaint ID"],
                "product": row["Product"]
            })
            all_chunks.append(chunk)

    print(f"Total chunks created: {len(all_chunks)}")

    # save chunks so we can embed them later
    import pickle
    os.makedirs(os.path.dirname(output_pickle), exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved text chunks to {output_pickle}")

if __name__ == "__main__":
    chunk_texts(
        input_csv="data/processed/sample_complaints.csv",
        output_pickle="data/processed/text_chunks.pkl"
    )
