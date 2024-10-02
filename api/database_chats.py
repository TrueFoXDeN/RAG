import os
from datetime import datetime, timezone

from pymongo import MongoClient

from api.schemas import ChatRequest

mongo_url = os.environ["MONGO_URL"]
mongo_user = os.environ["MONGO_USER"]
mongo_password = os.environ["MONGO_PASSWORD"]
client = MongoClient(
    f"mongodb://{mongo_user}:{mongo_password}@{mongo_url}:27017/?authSource=admin"
)
database = client["rag"]
collection = database["chats"]


def save_chat(messages: ChatRequest):
    collection.update_one(
        {"chat_id": messages.chat_id},
        {
            "$set": messages.dict(),
            "$setOnInsert": {"created_on": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
