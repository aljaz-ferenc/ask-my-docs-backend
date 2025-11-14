from fastapi import APIRouter, Depends
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
import os
from starlette.responses import JSONResponse
from app.core.schemas import AddFilesRequest, RemoveFilesRequest
import tempfile
from app.api.dependencies import get_file_storage
from app.api.dependencies import get_vector_store

files_router = APIRouter(prefix='/files', tags=['Files'])

@files_router.post('/')
async def add_files(req: AddFilesRequest, file_storage = Depends(get_file_storage), vector_store = Depends(get_vector_store)):
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


@files_router.delete('/')
async def remove_files(req: RemoveFilesRequest, vector_store = Depends(get_vector_store)):
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