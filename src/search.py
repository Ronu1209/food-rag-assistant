from qdrant_client import models
from .config import client, collection_name
from .embeddings import dense_embedding_model, sparse_embedding_model
from .state import RAGState

def search(state: RAGState) -> RAGState:
    query = state["query"]
    dense_vectors = next(dense_embedding_model.query_embed(query))
    sparse_vectors = next(sparse_embedding_model.query_embed(query))

    prefetch = [
        models.Prefetch(query=dense_vectors, using="dense", limit=15),
        models.Prefetch(query=models.SparseVector(**sparse_vectors.as_object()), using="sparse", limit=15)
    ]

    relevant_docs = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=dense_vectors,
        using="dense",
        with_payload=True,
        limit=3,
        query_filter=state.get("filter_conditions")
    )

    state["context"] = [p.payload["document"] for p in relevant_docs.points]
    return state
