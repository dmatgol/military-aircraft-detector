import io
import tempfile
import time
from typing import IO, Any

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

from model.model_utils import load_trained_model
from utils.utils import read_model_config

backend = " http://0.0.0.0:8080/"


def app_design():
    st.title("Military Aircraft detection Dashboard")
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider(
        "Confidence", min_value=0.5, max_value=1.0, value=0.85
    )
    image_movie = st.sidebar.selectbox("Image or Video Detection", ["image", "video"])
    instructions = """
        1. Please choose on the side bar between object detection on a video or an image. \
            Default will be an image.\n
        2. Based on your choice, you can either upload your own image \
        or select from the sidebar some cached images.
        """
    st.write(instructions)

    image_movie_selector = st.sidebar.selectbox(
        f"Upload image or select from cached {image_movie}",
        [f"Select cached {image_movie}", f"Upload new {image_movie}"],
    )
    if image_movie == "video":
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        if image_movie_selector == "Upload new video":
            file = st.sidebar.file_uploader(
                f"Upload an {image_movie}", type=["mp4", "mov", "avi", "m4v"]
            )
            if file:
                uploaded_video_design(tmp_file, file.read(), confidence)
        else:
            cached_video_design(tmp_file, image_movie, confidence)
    else:
        if image_movie_selector == "Upload new image":
            file = st.sidebar.file_uploader(
                f"Upload an {image_movie}", type=["jpeg", "jpg"]
            )
            if file:
                upload_image_design(file, confidence)

        else:
            cached_image_design(image_movie, confidence)


def uploaded_video_design(tmp_file: Any, file: IO, confidence: float):
    tmp_file.write(file)  # host file in a temp directory
    video = open(tmp_file.name, "rb")
    video_bytes = video.read()
    st.sidebar.title("Here is the video you've selected")
    st.sidebar.video(video_bytes)
    st.title("Here are the model predictions")
    stframe = st.empty()
    predict_video_frame(tmp_file.name, confidence, stframe)
    # Remove comment below if you want to generate video detections by calling the inference
    # API endoint. Video Inference will be slower.
    # video_detection_from_api(tmp_file, confidence)


def predict_video_frame(file_name: str, confidence: float, stframe):
    model_config = read_model_config("configs/model.yaml")
    model = load_trained_model(model_config)
    preds = model.predict_video(file_name, confidence, stframe)
    return preds


def cached_video_design(tmp_file: Any, file_type: str, confidence: float):
    files = get_cached_test_files_options(file_type)
    file_selector = st.sidebar.selectbox(f"Unseen {file_type}", files)
    movie = get_cached_test_file(file_type, file_selector)
    uploaded_video_design(tmp_file, movie, confidence)


def get_cached_test_files_options(file_type: str):
    url = backend + f"cache/{file_type}"
    data = {"file_type": file_type}
    r = requests.get(url=url, data=data)
    return r.json()["list_of_files"]


def get_cached_test_file(file_type: str, file_name: str):
    url = backend + f"cache/{file_type}/{file_name}"
    data = {"file_type": file_type, "file_name": file_name}
    r = requests.get(url=url, data=data)
    if file_type == "image":
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        return r.content


def upload_image_design(file: IO, confidence: float):
    image = Image.open(file)
    show_selected_image(image)
    preds = predict_image_from_uploaded_file(file, confidence)
    st.title("Here are the model predictions")
    st.image(preds, use_column_width=True)


def show_selected_image(image: Image):
    st.sidebar.title("Here is the image you've selected")
    resized_image = image.resize((256, 256))
    st.sidebar.image(resized_image)


@st.cache()
def predict_image_from_uploaded_file(file: Any, confidence: float):
    files = {"files": file.getvalue()}
    data = {"confidence": confidence}
    url = backend + "inference/image/uploaded_image"
    r = requests.get(url=url, data=data, files=files, timeout=8000)
    arr = np.frombuffer(r.content, np.uint8)
    annotated_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return annotated_image


def cached_image_design(file_type: str, confidence):
    file_selection = get_cached_test_files_options(file_type)
    file_selected = st.sidebar.selectbox(f"Unseen {file_type}", file_selection)
    image = get_cached_test_file(file_type, file_selected)
    show_selected_image(image)
    preds = predict_image_from_path(file_selected, confidence)
    st.title("Here are the model predictions")
    st.image(preds, use_column_width=True)


@st.cache()
def predict_image_from_path(filename_selected: str, confidence: float):
    url = backend + f"inference/image/image_from_path/{filename_selected}"
    data = {"image_name": filename_selected, "confidence": confidence}
    r = requests.get(url=url, data=data, timeout=8000)
    annotated_image = Image.open(io.BytesIO(r.content)).convert("RGB")
    return annotated_image


def video_detection_from_api(tmp_file, confidence):
    cap = cv2.VideoCapture(tmp_file.name)
    stframe = st.empty()
    while cap.isOpened():
        # Capture each frame of the video
        frame_return_value, frame = cap.read()
        if frame_return_value:

            # Compute prediction time to calculate frames per second
            start_time = time.time()
            frame = predict_video_frame_with_api(frame, confidence)

            end_time = time.time()

            fps = 1 / (end_time - start_time)
            from utils.utils import draw_fps

            frame = draw_fps(frame, fps)
            stframe.image(frame, use_column_width=True)
        else:
            break

    cap.release()


@st.cache()
def predict_video_frame_with_api(frame: np.array, confidence: float):
    image_frame = Image.fromarray(frame)
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        image_frame.save(buf, format="jpeg")
        im_bytes = buf.getvalue()
    frame_file = {"frame": im_bytes}
    data = {"confidence": confidence}
    url = backend + "inference/video/frame"
    r = requests.get(url=url, data=data, files=frame_file, timeout=8000)
    arr = np.frombuffer(r.content, np.uint8)
    img_np = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img_np is None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img_np
