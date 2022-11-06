import os

from api_models.api_model import FileType
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from settings.general import data_paths

router = APIRouter()


@router.get("/{file_type}")
def get_cached_files_options(file_type: FileType):
    if file_type == FileType.IMAGE:
        list_of_files = os.listdir(data_paths.test_images)
    elif file_type == FileType.VIDEO:
        list_of_files = os.listdir(data_paths.test_videos)
    else:
        raise HTTPException(
            status_code=404,
            detail="File type not correct. Please select: movie or image",
        )
    return {"list_of_files": list_of_files}


@router.get("/{file_type}/{file_name}", response_class=FileResponse)
def get_cached_file(file_type: str, file_name: str):
    if file_type == FileType.IMAGE.value:
        file_path = os.path.join(data_paths.test_images, file_name)
    elif file_type == FileType.VIDEO.value:
        file_path = os.path.join(data_paths.test_videos, file_name)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Image {file_name} not found. Please verify if name is correct.",
        )
    return FileResponse(file_path)
