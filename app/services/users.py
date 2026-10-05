from collections.abc import Mapping
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.models import User
from app.models.enums import UserRole
from app.schemas.user import GoogleUserInfo

# Only what is needed to call Gmail later; the id_token/userinfo are not persisted.
_PERSISTED_TOKEN_KEYS = ("access_token", "refresh_token", "token_type", "expires_at", "scope")


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_email(self, email: str) -> User | None:
        return await self._session.scalar(select(User).where(User.email == email))

    async def get(self, user_id: int) -> User | None:
        return await self._session.get(User, user_id)

    async def list_all(self) -> list[User]:
        return list(await self._session.scalars(select(User).order_by(User.id)))

    async def register_login(self, info: GoogleUserInfo, token: Mapping[str, Any]) -> User:
        """Create or update the user after a successful Google sign-in. The first user becomes admin."""
        user = await self.get_by_email(info.email)
        if user is None:
            user_count = await self._session.scalar(select(func.count()).select_from(User)) or 0
            user = User(
                email=info.email,
                role=(UserRole.ADMIN if user_count == 0 else UserRole.EMPLOYEE).value,
                active=True,
                created_at=utcnow(),
            )
            self._session.add(user)

        user.name = info.name or user.name
        user.picture = info.picture or user.picture
        user.last_login = utcnow()

        new_token = {k: token[k] for k in _PERSISTED_TOKEN_KEYS if k in token}
        # Google only returns a refresh token on consent; keep the previous one otherwise.
        if "refresh_token" not in new_token and user.google_token and "refresh_token" in user.google_token:
            new_token["refresh_token"] = user.google_token["refresh_token"]
        user.google_token = new_token

        await self._session.commit()
        return user

    async def set_role(self, user_id: int, role: UserRole) -> None:
        user = await self.get(user_id)
        if user is not None:
            user.role = role.value
            await self._session.commit()

    async def toggle_active(self, user_id: int) -> None:
        user = await self.get(user_id)
        if user is not None:
            user.active = not user.active
            await self._session.commit()
