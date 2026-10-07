"""
rag_app.py
Interactive UI for the from-scratch RAG system -- shows the retrieved chunks,
their cosine similarity scores, and the grounded answer.

Run:
    streamlit run rag_app.py
"""

import streamlit as st
from rag_from_scratch import build_knowledge_base, ask, ARTICLES

st.set_page_config(page_title="RAG From Scratch", page_icon="🔍", layout="centered")

st.title("🔍 RAG From Scratch")
st.caption("No LangChain, no managed vector DB — embeddings, cosine similarity, and generation, all built by hand, using Azure AI Foundry")

with st.expander(f"📚 Knowledge base: {len(ARTICLES)} articles"):
    for a in ARTICLES:
        st.markdown(f"**{a['title']}**")

if "index" not in st.session_state:
    if st.button("⚙️ Build Index (embed all chunks)"):
        with st.spinner("Chunking and embedding articles via Azure AI Foundry..."):
            st.session_state.index = build_knowledge_base()
        st.success(f"Index built: {len(st.session_state.index)} chunks embedded.")
        st.rerun()
else:
    st.success(f"✅ Index ready: {len(st.session_state.index)} chunks embedded.")

    query = st.text_input("Ask a question about the knowledge base:")

    if query:
        with st.spinner("Retrieving and generating..."):
            result = ask(query, st.session_state.index, k=3, verbose=False)

        st.subheader("Retrieved Chunks (by cosine similarity)")
        for r in result["retrieved"]:
            with st.expander(f"[{r['score']:.3f}] {r['title']}"):
                st.write(r["chunk_text"])

        st.subheader("Grounded Answer")
        st.write(result["answer"])
