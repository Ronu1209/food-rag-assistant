from langchain_google_genai import ChatGoogleGenerativeAI
from .state import RAGState

def answer(state: RAGState) -> RAGState:
    context = " ".join(state["context"])
    query = state["query"]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    prompt = f"""
You are a seasoned master chef working at five star hotel - Taj Hotels. 
Use ONLY the CONTEXT to answer. If not found, say: "I don't know, not enough information provided."

<context>
{context}
</context>

<question>
{query}
</question>
"""

    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state
