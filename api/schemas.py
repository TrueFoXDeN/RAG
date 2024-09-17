from enum import Enum
from typing import List

from pydantic import BaseModel


class LanguageEnum(str, Enum):
    english = "english"
    german = "german"


class EmbedRequest(BaseModel):
    text: List[str]
