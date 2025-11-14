from functools import lru_cache
from app.services.file_storage import FileStorage
from app.services.vector_store import VectorStore

@lru_cache
def get_vector_store():
    return VectorStore('ask-my-docs')

@lru_cache
def get_file_storage():
    return FileStorage()