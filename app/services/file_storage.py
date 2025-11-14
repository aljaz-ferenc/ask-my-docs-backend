from appwrite.client import Client
from appwrite.services.storage import Storage
import os
from dotenv import load_dotenv

load_dotenv()

class FileStorage:
    def __init__(self):
        self.client = Client()
        self.client.set_endpoint(os.getenv('APPWRITE_ENDPOINT'))
        self.client.set_project(os.getenv('APPWRITE_PROJECT_ID'))
        self.storage = Storage(self.client)

    def get_file_metadata(self, file_id: str):
        return self.storage.get_file(
            bucket_id=os.getenv("APPWRITE_BUCKET_ID"),
            file_id=file_id,
        )

    def download_file(self, file_id: str) -> bytes:
        return self.storage.get_file_download(
            bucket_id=os.getenv("APPWRITE_BUCKET_ID"),
            file_id=file_id
        )


