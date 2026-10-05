from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import UserRole
from app.schemas.common import DisplayStr


class GoogleUserInfo(BaseModel):
    """Subset of the OpenID Connect `userinfo` claims returned by Google."""

    model_config = ConfigDict(extra="ignore")

    email: str
    name: str | None = None
    picture: str | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: DisplayStr
    picture: str | None
    role: UserRole
    active: bool
    created_at: datetime | None
    last_login: datetime | None


class RoleUpdateForm(BaseModel):
    role: UserRole
