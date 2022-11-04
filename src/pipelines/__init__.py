from enum import Enum


class RunMode(str, Enum):
    TRAIN = "train"
    INFERENCE = "inference"


__all__ = ["RunMode"]
