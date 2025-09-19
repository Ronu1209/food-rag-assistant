from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from src.store import client, collection_name, create_collection_safe, dense_embedding_model, sparse_embedding_model
from qdrant_client import models
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Initialize FastAPI
app = FastAPI(title="Recipe RAG Assistant", version="1.0")

# Ensure collection exists
create_collection_safe()


# -----------------------------
# Define RAG State
# -----------------------------
class RAGState(TypedDict):
    query: str
    context: List[str]
    answer: str
    filter_conditions: Optional[Dict[str, Any]]


# -----------------------------
# Search Function
# -----------------------------
def search(state: RAGState) -> RAGState:
    query = state["query"]

    # Embed query
    dense_vectors = next(dense_embedding_model.query_embed(query))
    sparse_vectors = next(sparse_embedding_model.query_embed(query))

    prefetch = [
        models.Prefetch(query=dense_vectors, using="dense", limit=10),
        models.Prefetch(query=models.SparseVector(**sparse_vectors.as_object()), using="sparse", limit=10),
    ]

    query_filter = state.get("filter_conditions")

    relevant_docs = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=dense_vectors,
        using="dense",
        with_payload=True,
        limit=3,
        query_filter=query_filter,
    )

    context = [point.payload["document"] for point in relevant_docs.points]

    state["context"] = context
    return state


# -----------------------------
# Answer Function
# -----------------------------
def answer(state: RAGState) -> RAGState:
    context = " ".join(state["context"])
    query = state["query"]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_tokens=None)

    prompt = f"""
You are a cooking assistant. Answer based only on the CONTEXT below.
If the answer is not in the context, say: "I don't know, not enough information provided."

<context>
{context}
</context>

<question>
{query}
</question>
"""
    answer_text = llm.invoke(prompt)
    state["answer"] = answer_text.content
    return state


# -----------------------------
# Build Workflow Graph
# -----------------------------
workflow = StateGraph(RAGState)
workflow.add_node("search_context", search)
workflow.add_node("answer_generation", answer)
workflow.add_edge(START, "search_context")
workflow.add_edge("search_context", "answer_generation")
workflow.add_edge("answer_generation", END)
graph = workflow.compile()


# -----------------------------
# API Models
# -----------------------------
class QueryIn(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None


class QueryOut(BaseModel):
    answer: str


# -----------------------------
# Routes
# -----------------------------
@app.post("/query", response_model=QueryOut)
def query_recipe(in_data: QueryIn):
    """Ask a recipe-related question."""
    result = graph.invoke({
        "query": in_data.query,
        "context": [],
        "answer": "",
        "filter_conditions": in_data.filters,
    })
    return {"answer": result["answer"]}


@app.get("/")
def root():
    return {"message": "Welcome to the Recipe RAG Assistant API!"}
