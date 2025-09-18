from src.loaders import load_documents, split_documents
from src.store import create_collection_safe, process_and_store_documents

if __name__ == "__main__":
    print("📖 Loading recipe book...")
    docs = load_documents()
    chunks = split_documents(docs)

    print("🍳 Creating collection in Qdrant...")
    create_collection_safe()

    print("🥗 Storing recipe chunks...")
    process_and_store_documents(chunks)

    print("✅ Ingestion complete. You can now start the API server.")
