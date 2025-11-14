import chromadb
from chromadb.errors import ChromaError
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4

load_dotenv()

class VectorStore:
    def __init__(self, collection_name):
        self.collection_name = collection_name

        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(db_path, exist_ok=True)
        print("DB_PATH: ", db_path)
        self.client = chromadb.PersistentClient(path=db_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # if collection_name in [c.name for c in self.client.list_collections()]:
        #     self.client.delete_collection(collection_name)

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction(
                    model_name="text-embedding-3-small",
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            )
        except ChromaError as e:
            print(f"Error initializing collection: {e}")
            raise


    def add_docs(self, docs, ids=None):
        if ids is None:
            ids = [str(uuid4()) for _ in docs]
        self.collection.upsert(
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
            ids=ids
        )
        print(f"Added {len(docs)} docs to database.")

    def split_text(self, docs):
        chunks = self.text_splitter.split_documents(docs)
        print(f"Split docs into {len(chunks)} chunks.")
        return chunks

    def query(self, query_texts, n_results=3):
        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
        formatted = [
            {
                "text": doc,
                "metadata": meta,
                "score": score
            }
            for doc, meta, score in zip(
                results['documents'][0],
                results.get('metadatas', [[]])[0],
                results.get('distances', [])[0]
            )
        ]
        return formatted

    def reset_collection(self):
        self.client.delete_collection(self.collection_name)