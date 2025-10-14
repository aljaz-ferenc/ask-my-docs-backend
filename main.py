from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from dotenv import load_dotenv
import os
from starlette.responses import JSONResponse
from models.models import QueryRequest, AddFilesRequest, RemoveFilesRequest
from file_storage import FileStorage
from llms.chat_model import run_chat_model
from vector_store import VectorStore
import tempfile

load_dotenv()

origin = os.getenv('ORIGIN')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin],
    allow_credentials=True,
    allow_methods=["GET", 'POST'],
    allow_headers=["*"],
)

vector_store = VectorStore('ask-my-docs')
file_storage = FileStorage()

@app.post("/query")
async def query(req: QueryRequest):
    results = vector_store.query([req.query])
    print(f"Found {len(results)} results for query: {req.query}")

    texts = [result["text"] for result in results]
    context = ", ".join(texts)

    response = run_chat_model(context=context, query=req.query, recent_messages=req.recentMessages)
    print(response)

    return {"results": results, "llm_response": response.choices[0].message.content}


@app.post('/files')
async def add_files(req: AddFilesRequest):
    try:
        docs = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for file_id in req.filesIds:
                file_metadata = file_storage.get_file_metadata(file_id)
                file_name = file_metadata['name']

                file_bytes = file_storage.download_file(file_id)
                temp_path = os.path.join(temp_dir, file_name)
                with open(temp_path, 'wb') as f:
                    f.write(file_bytes)

                if file_name.endswith('.pdf'):
                    loader = PyMuPDFLoader(temp_path)
                    file_docs = loader.load()
                elif file_name.endswith('.txt'):
                    loader = TextLoader(temp_path)
                    file_docs = loader.load()
                else:
                    print(f"Skipping unsupported file type: {file_name}")
                    continue

                for doc in file_docs:
                    doc.metadata['file_id'] = file_id
                    doc.metadata['file_name'] = file_name

                docs.extend(file_docs)

        chunks = vector_store.split_text(docs)
        vector_store.add_docs(chunks)

        return {"status": "success", "fileId": req.filesIds[0]}

    except Exception as e:
        print(f"Error adding documents: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete('/files')
async def remove_files(req: RemoveFilesRequest):
    deleted_count = 0

    for file_id in req.filesIds:
        try:
            vector_store.collection.delete(
                where={"file_id": file_id}
            )
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete file: {e}")

    print(f"Deleted {deleted_count} files.")
    return {'status': 'success',  "fileId": req.filesIds[0]}