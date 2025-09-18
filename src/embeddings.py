from fastembed import TextEmbedding, SparseTextEmbedding

dense_embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
sparse_embedding_model = SparseTextEmbedding("Qdrant/BM25")
