# src/rag_pipeline.py

from transformers import pipeline
from src.retriever import retrieve_documents
from src.prompt import build_prompt

# Load a generation model



print("Loading Mistral model — this may take several minutes on CPU …")
from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=300
    print("Model loaded!")
)





def run_rag(question: str, k: int = 5):
    # 1. Retrieve relevant text chunks
    docs = retrieve_documents(question, k=k)

    # 2. Build a single context string
    context = "\n\n".join([f"- {d.page_content}" for d in docs])

    # 3. Build the prompt
    prompt = build_prompt(context, question)

    # 4. Generate answer
    output = generator(prompt)
    text = output[0]["generated_text"]

    return {
        "answer": text,
        "sources": docs
    }
