from typing import Literal
from pydantic import BaseModel

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