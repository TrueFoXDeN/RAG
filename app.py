import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import routes

logging.basicConfig(level=logging.DEBUG)

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})

origins = [
    "http://localhost",
    "http://localhost:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)
