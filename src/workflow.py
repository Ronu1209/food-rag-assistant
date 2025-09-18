from langgraph.graph import StateGraph, START, END
from .state import RAGState
from .search import search
from .answer import answer

def build_workflow():
    workflow = StateGraph(RAGState)
    workflow.add_node("search_context", search)
    workflow.add_node("answer_generation", answer)
    workflow.add_edge(START, "search_context")
    workflow.add_edge("search_context", "answer_generation")
    workflow.add_edge("answer_generation", END)
    return workflow.compile()
