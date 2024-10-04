import os
from datetime import datetime, timezone

from pymongo import MongoClient

from api.schemas import ChatRequest, MessageRequest

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
    )


def save_message(message: MessageRequest):
    collection.update_one(
        {"chat_id": message.chat_id},
        {
            "$push": {"messages": message.dict(exclude={"chat_id"})},
        },
    )


def get_chat(chat_id):
    return collection.find_one({"chat_id": chat_id})


def get_all_chats():
    chats = collection.find(
        {},  # Keine Filter, also alle Dokumente
        {
            "chat_id": 1,
            "summary": 1,
            "created_on": 1,
            "_id": 0,
        },  # Nur diese Felder zurückgeben, _id ausschließen
    ).sort(
        "created_on", 1
    )  # Sortierung nach created_on, 1 für aufsteigend, -1 für absteigend

    return list(chats)
