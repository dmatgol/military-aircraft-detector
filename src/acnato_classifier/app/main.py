import socket

import numpy as np
import uvicorn
from _version import __version__
from app.api_models.response_model import APIState, Model
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()


@app.get("/", response_model=APIState)
def heartbeat():
    """Returns state of API"""
    print("heartbeat queried")
    return APIState(machine_name=socket.gethostname(), version=__version__)


@app.post("/predict")
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))  # noqa
    return "Implement Predict logic"


@app.post("/train/")
def train_model(model: Model):
    return f"Implement train for {model} logic"


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
