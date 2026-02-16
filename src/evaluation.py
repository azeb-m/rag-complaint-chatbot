# src/evaluation.py

from src.rag_pipeline import run_rag

questions = [
    "Why are customers unhappy with credit cards?",
    "What issues do customers report about savings accounts?",
    "Are there common money transfer delays?",
    "What problems do users report about personal loans?",
    "Which company has the most complaint mentions?"
]

def evaluate():
    results = []
    for q in questions:
        output = run_rag(q, k=5)

        results.append({
            "question": q,
            "answer": output["answer"],
            "sources": [s.page_content[:200] for s in output["sources"]]
        })
    return results

if __name__ == "__main__":
    for r in evaluate():
        print("\nQuestion:", r["question"])
        print("Answer:", r["answer"])
        print("Source Example:", r["sources"][0])
