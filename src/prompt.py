# src/prompt.py

def build_prompt(context: str, question: str) -> str:
    return f"""
You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer customer feedback questions using ONLY the retrieved complaint excerpts below.
If the context does NOT contain enough information to answer the question, state that clearly.

Context:
{context}

Question:
{question}

Answer:
""".strip()
