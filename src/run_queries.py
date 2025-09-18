from .loaders import load_documents, split_documents
from .store import create_collection, process_and_store_documents
from .workflow import build_workflow

# Step 1. Load & split documents
docs = load_documents()
chunks = split_documents(docs)

# Step 2. Store in Qdrant
create_collection()
process_and_store_documents(chunks)

# Step 3. Build workflow
graph = build_workflow()

def query_system(question, filters=None):
    return graph.invoke({
        "query": question,
        "context": [],
        "answer": "",
        "filter_conditions": filters
    })["answer"]

# Test queries
queries = [
    "What are the main ingredients in a Vegetable Spring Rolls?",
    "What is the method to make Potato Chat?",
    "Recipe for Indo chinese bhel"  # should return I don't know
]

for q in queries:
    print("Q:", q)
    print("A:", query_system(q))
    print("-"*60)
