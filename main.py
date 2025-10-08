from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from dotenv import load_dotenv
import os

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

    print(f"Loaded {len(docs)} documents.")
    return {'message': 'hello'}




