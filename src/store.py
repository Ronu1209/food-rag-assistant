import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from fastembed import TextEmbedding, SparseTextEmbedding

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing. Please set it in your .env file.")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collection_name = "indian_recipes"

# Embedding models
dense_embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
sparse_embedding_model = SparseTextEmbedding("Qdrant/BM25")


def create_collection_safe():
    """Create the Qdrant collection if it does not already exist."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE, on_disk=True)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        print(f"✅ Created collection `{collection_name}`")
    except UnexpectedResponse as e:
        if "already exists" in str(e):
            print(f"⚠️ Collection `{collection_name}` already exists, skipping creation.")
        else:
            raise


def process_and_store_documents(chunks):
    """Embed and upload document chunks to Qdrant (auto-delete old ones)."""

    # ⚠️ Delete old points first
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=models.Filter(must=[]))  # deletes all
    )
    print(f"🗑️ Cleared old data from `{collection_name}`")

    points = []

    # Create embeddings
    dense_embeddings = list(dense_embedding_model.embed([doc.page_content for doc in chunks]))
    sparse_embeddings = list(sparse_embedding_model.embed([doc.page_content for doc in chunks]))

    for idx, (chunk, dense_emb, sparse_emb) in enumerate(
        zip(chunks, dense_embeddings, sparse_embeddings)
    ):
        point = models.PointStruct(
            id=idx,
            vector={
                "dense": dense_emb,
                "sparse": sparse_emb.as_object(),
            },
            payload={
                "document": chunk.page_content,
                "source": chunk.metadata.get("source", "recipes_book.pdf"),
            },
        )
        points.append(point)

    # Upload to Qdrant
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Stored {len(points)} chunks in `{collection_name}`")
