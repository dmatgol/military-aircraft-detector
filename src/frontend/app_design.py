import os
import tempfile

import streamlit as st
from PIL import Image

from model.model_utils import load_trained_model
from settings.general import data_paths


def app_design(model_config):
    model = load_model(model_config)
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

    warning_indication = """
       <link rel="stylesheet" \
           href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

        <i class="fa-solid fa-triangle-exclamation"></i>  \
            **No military aircraft detected with this level of confidence.**
    """

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
                tmp_file.write(file.read())
                video = open(tmp_file.name, "rb")
                video_bytes = video.read()
                st.sidebar.title("Here is the video you've selected")
                st.sidebar.video(video_bytes)
                stframe = st.empty()
                predict_video(tmp_file.name, model, confidence, stframe)
        else:
            file_location = get_test_file(data_paths.test_videos, image_movie)
            video = open(file_location, "rb")
            video_bytes = video.read()
            st.sidebar.title("Here is the video you've selected")
            st.sidebar.video(video_bytes)
            st.title("Here are the model predictions")
            stframe = st.empty()
            predict_video(file_location, model, confidence, stframe)
    else:
        if image_movie_selector == "Upload new image":
            file = st.sidebar.file_uploader(f"Upload an {image_movie}")
            if file:
                show_selected_image(file)
                preds, object_detected = predict_image(file, model, confidence)
                st.title("Here is the model predictions")
                if not object_detected:
                    st.write(warning_indication, unsafe_allow_html=True)
                st.image(preds, use_column_width=True)
        else:
            file_location = get_test_file(data_paths.test_images, image_movie)
            show_selected_image(file_location)
            preds, object_detected = predict_image(file_location, model, confidence)
            st.title("Here are the model predictions")
            if not object_detected:
                st.write(warning_indication, unsafe_allow_html=True)
            st.image(preds, use_column_width=True)


def get_test_file(file_location: str, image_movie: str):
    file = os.listdir(file_location)
    file_selector = st.sidebar.selectbox(f"Unseen {image_movie}", file)
    file_location = os.path.join(file_location, file_selector)
    return file_location


def show_selected_image(file):
    img = Image.open(file)

    st.sidebar.title("Here is the image you've selected")
    resized_image = img.resize((256, 256))
    st.sidebar.image(resized_image)


@st.cache(allow_output_mutation=True)
def load_model(model_config):
    model = load_trained_model(model_config)
    return model


@st.cache()
def predict_image(file_location: str, model, confidence: float):
    preds, object_detected = model.predict_image(file_location, confidence)
    return preds, object_detected


@st.cache()
def predict_video(file_location: str, model, confidence: float, stframe):
    preds = model.predict_video(file_location, confidence, stframe)
    return preds
