from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from dotenv import load_dotenv
import os

from starlette.responses import JSONResponse

from vector_store import VectorStore

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

@app.post('/upload')
async def upload(files: List[UploadFile] = File(...)):
    docs = []
    for file in files:
        if file.filename.endswith('.pdf'):
            contents = await file.read()
            with open(file.filename, 'wb') as f:
                f.write(contents)

            loader = PyMuPDFLoader(file.filename)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)

        elif file.filename.endswith('.txt'):
            with open(file.filename, 'wb') as f:
                f.write(await file.read())

            loader = TextLoader(file.filename)
            text_docs = loader.load()
            docs.extend(text_docs)

    try:
        chunks = vector_store.split_text(docs)
        vector_store.add_docs(chunks)
        return {'message': f"Processed {len(docs)} documents."}
    except Exception as e:
        print(f"Upload error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


