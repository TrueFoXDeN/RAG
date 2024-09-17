from fastapi import FastAPI

from api import routes

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})

app.include_router(routes.router)
