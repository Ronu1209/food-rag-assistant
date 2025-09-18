from typing import TypedDict, List, Dict, Optional, Any

class RAGState(TypedDict):
    query: str
    context: List[str]
    answer: str
    filter_conditions: Optional[Dict[str, Any]]
