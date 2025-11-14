from fastapi import APIRouter, Depends
from app.core.schemas import QueryRequest
from app.services.chat_model import run_chat_model
from app.api.dependencies import get_vector_store

query_router = APIRouter(prefix='/query', tags=['Query'])

@query_router.post("/")
async def query(req: QueryRequest, vector_store = Depends(get_vector_store)):
    results = vector_store.query([req.query])
    print(f"Found {len(results)} results for query: {req.query}")

    texts = [result["text"] for result in results]
    context = ", ".join(texts)

    response = run_chat_model(context=context, query=req.query, recent_messages=req.recentMessages)
    print(response)

    return {"results": results, "llm_response": response.choices[0].message.content}


