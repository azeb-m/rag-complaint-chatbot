# src/streamlit_app.py

import streamlit as st
from src.rag_pipeline import run_rag

st.set_page_config(page_title="Intelligent Complaint Chatbot", layout="wide")

st.title("📊 Intelligent Complaint Analysis Chatbot")
st.write("Ask questions about consumer complaints and get evidence-backed answers.")

question = st.text_input("Ask your question:")

if st.button("Submit") and question:
    with st.spinner("Generating answer…"):
        result = run_rag(question, k=5)

    st.subheader("🔍 Answer")
    st.write(result["answer"])

    st.subheader("📄 Sources")
    for idx, doc in enumerate(result["sources"], start=1):
        st.markdown(f"**Source {idx}:** {doc.page_content[:300]}...")
