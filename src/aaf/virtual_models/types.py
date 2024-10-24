import time
from typing import Literal

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    id: str
    name: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = ""


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = []
