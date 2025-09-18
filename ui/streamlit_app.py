import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"  # FastAPI backend

st.set_page_config(page_title="Recipe RAG Assistant", page_icon="🍲")

st.title("🍲 Recipe RAG Assistant")
st.write("Ask me anything about the recipe book!")

# Input box
query = st.text_area("Enter your question:")

# Optional: allow filters
use_filter = st.checkbox("Use filter (advanced)")
filters = {}
if use_filter:
    key = st.text_input("Filter key (e.g., 'source')")
    val = st.text_input("Filter value (e.g., 'recipes_book.pdf')")
    if key and val:
        filters[key] = val

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Searching recipes..."):
            try:
                payload = {"query": query, "filters": filters or None}
                res = requests.post(API_URL, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.subheader("Answer")
                    st.write(data["answer"])
                else:
                    st.error(f"API error: {res.status_code} {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
st.markdown("---")
st.success("✅ Streamlit UI loaded successfully!")