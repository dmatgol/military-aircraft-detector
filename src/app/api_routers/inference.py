import io
import os

from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image
from starlette.responses import Response

from src.app.model.model_utils import load_trained_model
from src.app.settings.general import data_paths
from src.app.utils.utils import read_model_config

router = APIRouter()


def load_model():
    model_config = read_model_config("configs/model.yaml")
    model = load_trained_model(model_config)
    return model


@router.post("/image/uploaded_image/")
async def get_predicted_bbox_from_uploaded_file(
    files: UploadFile = File(...), confidence: float = Form(...)
):
    image = files.file.read()
    model = load_model()
    preds, _ = model.predict_image(image, confidence)
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(preds)
    img_base64.save(bytes_io, format="jpeg")
    return Response(bytes_io.getvalue())


@router.post("/image/image_from_path/{image_name}")
async def get_predicted_bbox_from_image_path(
    image_name: str, confidence: float = Form(...)
):
    file_path = os.path.join(data_paths.test_images, image_name)
    model = model = load_model()
    preds, _ = model.predict_image(file_path, confidence)
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(preds)
    img_base64.save(bytes_io, format="jpeg")
    return Response(bytes_io.getvalue())


@router.post("/video/frame")
async def get_predicted_bbox_from_video_frame(
    frame: UploadFile = File(...), confidence: float = Form(...)
):
    frame = frame.file.read()
    model = load_model()
    preds, _ = model.predict_image(frame, confidence)
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(preds)
    img_base64.save(bytes_io, format="jpeg")
    return Response(bytes_io.getvalue())
