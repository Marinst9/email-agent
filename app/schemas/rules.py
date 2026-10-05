from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from app.schemas.common import DisplayStr


class ReplyTemplateCreate(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    keywords: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    response: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ReplyTemplateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: DisplayStr
    keywords: list[str]
    response: DisplayStr

    @field_validator("keywords", mode="before")
    @classmethod
    def split_keywords(cls, value: Any) -> Any:
        # Stored as a comma-separated string. Blank entries are dropped: an empty
        # keyword would otherwise match every email.
        if value is None:
            return []
        if isinstance(value, str):
            return [k.strip() for k in value.split(",") if k.strip()]
        return value


class BlockedWordForm(BaseModel):
    word: Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=100)]
