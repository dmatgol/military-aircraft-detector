from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class FileType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class BaseModelExternal(BaseModel):
    def __str__(self):
        return str({x: y for x, y in self.__dict__.items() if y is not None})


class Model(str, Enum):
    basemodel = "basemodel"


class APIState(BaseModelExternal):
    machine_name: str
    version: str
