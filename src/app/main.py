import socket

import uvicorn
from fastapi import FastAPI

from src.app.api_models.api_model import APIState
from src.app.api_routers import cached_files, inference

app = FastAPI(
    title="Military Aircraft Detector",
    description="""Detect military aircraft in images or in video.""",
    version="0.1.0",
)


app.include_router(inference.router, prefix="/inference")
app.include_router(cached_files.router, prefix="/cache")


@app.get("/", response_model=APIState)
def heartbeat():
    """Returns state of API"""
    print("heartbeat queried")
    return APIState(machine_name=socket.gethostname(), version=app.version)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
