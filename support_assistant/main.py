import os
from typing import TypedDict, List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

app = FastAPI(title="Zepto Support RAG API")

class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

docs = [
    "Zepto guarantees delivery within 10 minutes for active service zones.",
    "Order cancellations are allowed within 60 seconds of placing the order.",
    "For damaged items, refunds are processed within 24-48 hours back to original payment source."
]

vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def retrieve(state: GraphState):
    q = state["question"]
    retrieved_docs = retriever.invoke(q)
    doc_texts = [d.page_content for d in retrieved_docs]
    return {"documents": doc_texts, "question": q}

def generate(state: GraphState):
    q = state["question"]
    context = "\n".join(state["documents"])
    prompt = f"Answer the user query based only on context:\nContext: {context}\nQuery: {q}"
    response = llm.invoke(prompt)
    return {"generation": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_graph = workflow.compile()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    result = app_graph.invoke({"question": req.question})
    return QueryResponse(
        question=req.question,
        answer=result["generation"]
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
