import streamlit as st
import requests
import os
from dotenv import load_dotenv
from main import graph

load_dotenv()


st.set_page_config(page_title="Recipe RAG Assistant", page_icon="🍲")

st.title("🍲 Recipe RAG Assistant")
st.write("Ask me anything about the Food Recipes!")

# Input box
query = st.text_area("Enter your question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Searching recipes..."):
            try:
                payload = {"query": query}
                response = graph.invoke(payload)
                if response["answer"]:
                    st.subheader("Answer")
                    st.write(response["answer"])
                else:
                    pass
            except Exception as e:
                st.error(f"Connection error: {e}")
st.markdown("---")
st.success("✅ Streamlit UI loaded successfully!")