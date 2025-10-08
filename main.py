from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from starlette.responses import JSONResponse
from vector_store import VectorStore
import tempfile
from openai import OpenAI

class QueryRequest(BaseModel):
    query: str

load_dotenv()

origin = os.getenv('ORIGIN')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin],  # your frontend
    allow_credentials=True,
    allow_methods=["GET", 'POST'],  # important for OPTIONS preflight
    allow_headers=["*"],
)

vector_store = VectorStore('ask-my-docs')
openai_client = OpenAI()

@app.post('/upload')
async def upload(files: List[UploadFile] = File(...)):
    docs = []

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                temp_path = os.path.join(temp_dir, file.filename)

                contents = await file.read()
                with open(temp_path, "wb") as f:
                    f.write(contents)

                if file.filename.endswith(".pdf"):
                    loader = PyMuPDFLoader(temp_path)
                    pdf_docs = loader.load()
                    docs.extend(pdf_docs)
                elif file.filename.endswith(".txt"):
                    loader = TextLoader(temp_path)
                    text_docs = loader.load()
                    docs.extend(text_docs)

        chunks = vector_store.split_text(docs)
        vector_store.add_docs(chunks)

        return {"message": f"Processed {len(docs)} documents."}

    except Exception as e:
        print(f"Upload error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/query")
async def query(req: QueryRequest):
    results = vector_store.query([req.query])
    print(f"Found {len(results)} results for query: {req.query}")

    texts = [result["text"] for result in results]
    context = ", ".join(texts)

    system_prompt = """
        You are a helpful assistant. You are given a context and a question. 
        Answer the question **using only information from the context**. 
        Do not make up anything. 
        Keep your answer **short, concise, and to the point**, ideally 1–2 sentences. 
        If the answer cannot be found in the context, say: 
        'I cannot find the information in the provided files.'
    """

    user_prompt = f"""
    Context: {context}.
    \n\nQuestion: {req.query}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        max_tokens=150
    )
    print(response)

    return {"results": results, "llm_response": response.choices[0].message.content}