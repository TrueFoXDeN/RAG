from enum import Enum
from typing import List

from pydantic import BaseModel


class LanguageEnum(str, Enum):
    english = "english"
    german = "german"


class EmbedRequest(BaseModel):
    text: List[str]


class QueryRequest(BaseModel):
    query: str


class MessageType(str, Enum):
    prompt = "prompt"
    response = "response"


class Context(BaseModel):
    file: str
    page: int
    text: str


class Message(BaseModel):
    context: List[Context]
    text: str
    type: MessageType

    class Config:
        use_enum_values = True


class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]
