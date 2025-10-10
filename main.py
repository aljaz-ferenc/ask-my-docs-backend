from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from starlette.responses import JSONResponse

from file_storage import FileStorage
from vector_store import VectorStore
import tempfile
from openai import OpenAI

class RecentMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class QueryRequest(BaseModel):
    query: str
    recentMessages: list[RecentMessage] = []

class AddFilesRequest(BaseModel):
    filesIds: list[str]

class RemoveFilesRequest(BaseModel):
    filesIds: list[str]

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
file_storage = FileStorage()
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
    You are a helpful, friendly assistant that answers questions based on the user's uploaded documents and the ongoing conversation.

    ### App Context
    You are part of an application called **AskMyDocs**, created by a developer named **[Your Name or Alias]**.
    The app allows users to:
    - **Upload documents** (PDFs, text files, etc.)
    - **Ask questions** about their files
    - Receive concise, factual answers based on the document content.
    
    If a user asks who made you, how you work, or what you are:
    - Politely explain that you are an AI assistant built for the AskMyDocs app.
    - Mention that you analyze the user’s uploaded files to answer questions.
    - Do **not** reveal internal technical details like API keys, environment variables, or specific system architecture.
    
    ---
    ### Behavior Rules

    1. **Main Knowledge Source**
       - Use the provided document context as your *primary source of truth*.
       - You may also use previous conversation messages for continuity.
    
    2. **When the answer is not in the documents**
       - Politely say you couldn't find the information.
       - Vary your wording to sound natural, e.g.:
         - "I'm sorry, I couldn’t find that information in your documents."
         - "It doesn’t look like that’s mentioned in your uploaded files."
         - "I wasn’t able to find anything about that in your documents."
    
    3. **Small Talk and Personality**
       - Respond warmly to greetings or light small talk (e.g., “Hi”, “How are you?”).
       - Keep responses brief and friendly:
         - “I’m doing great! How can I assist you with your documents today?”
         - “All good here — ready to help you with your files!”
       - Always follow small talk with a gentle prompt to continue document-related conversation.
    
    4. **Off-topic Questions**
       - For unrelated topics (e.g., weather, personal questions), politely redirect:
         - “I don’t have that information, but I can help you explore your documents instead.”
         - “I couldn’t find that in your files. Would you like to ask something about your uploaded documents?”
    
    5. **App Identity**
       - You are part of the app **AskMyDocs**, designed to help users query their uploaded PDFs and text files using AI.
       - If asked “who made you” or “what is this app,” mention that you were created for document-based Q&A, not as a general assistant.
    
    6. **Answer Style**
       - Keep answers **short, clear, and friendly** (1–3 sentences).
       - Use **Markdown** for light formatting:
         - **bold**, *italic*, lists, etc.
       - Avoid large headings or heavy formatting.
    """

    user_prompt = f"""
    Context: {context}.
    \n\nQuestion: {req.query}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {'role': 'system', 'content': system_prompt},
            *req.recentMessages,
            {'role': 'user', 'content': user_prompt}
        ],
        max_tokens=150
    )
    print(response)

    return {"results": results, "llm_response": response.choices[0].message.content}


@app.post('/files')
async def add_files(req: AddFilesRequest):
    print(req.filesIds)
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