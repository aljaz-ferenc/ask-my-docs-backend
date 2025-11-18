import asyncio
from fastapi import APIRouter, Depends, Query
from app.services.chat_model import run_chat_model
from app.api.dependencies import get_vector_store
from sse_starlette.sse import EventSourceResponse
import json

query_router = APIRouter(prefix='/query', tags=['Query'])

@query_router.get("/")
async def llm_stream(
    query: str = Query(default=""),
    recent_messages: str = Query(default="[]"),
    vector_store = Depends(get_vector_store)
):
    results = vector_store.query([query])
    print(f"Found {len(results)} results for query: {query}")
    metadata = [result['metadata'] for result in results]

    texts = [result["text"] for result in results]
    context = ", ".join(texts)

    async def event_generator():
        yield {'event': 'metadata', 'data': json.dumps(metadata)}

        for token in run_chat_model(context, query, recent_messages):
            yield {'event': 'token', 'data': token}
            await asyncio.sleep(0.05)
            
        yield {"event": "done", "data": "done"}

    return EventSourceResponse(event_generator())

