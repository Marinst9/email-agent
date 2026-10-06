"""Admins cannot deactivate or demote themselves (they could lock everyone out of the admin panel)."""

from typing import Any, cast

import pytest
from starlette.requests import Request

from app.api.routers.admin import change_role, router, toggle_user
from app.core.errors import PermissionDeniedError
from app.models import User
from app.models.enums import UserRole
from app.schemas.user import RoleUpdateForm


class FakeUsers:
    def __init__(self) -> None:
        self.role_changes: list[tuple[int, UserRole]] = []
        self.toggled: list[int] = []

    async def set_role(self, user_id: int, role: UserRole) -> None:
        self.role_changes.append((user_id, role))

    async def toggle_active(self, user_id: int) -> None:
        self.toggled.append(user_id)


def _request() -> Request:
    app = type("App", (), {"url_path_for": router.url_path_for})()
    return Request({"type": "http", "app": app, "method": "POST", "path": "/", "headers": [], "query_string": b""})


ADMIN = User(id=1, email="admin@firma.mk", role=UserRole.ADMIN.value, active=True)


@pytest.mark.parametrize("role", [UserRole.MANAGER, UserRole.EMPLOYEE])
async def test_admin_cannot_demote_themselves(role: UserRole) -> None:
    users = FakeUsers()
    with pytest.raises(PermissionDeniedError):
        await change_role(_request(), ADMIN.id, RoleUpdateForm(role=role), ADMIN, cast(Any, users))
    assert users.role_changes == []


async def test_admin_cannot_deactivate_themselves() -> None:
    users = FakeUsers()
    with pytest.raises(PermissionDeniedError):
        await toggle_user(_request(), ADMIN.id, ADMIN, cast(Any, users))
    assert users.toggled == []


async def test_admin_can_still_manage_other_users() -> None:
    users = FakeUsers()
    await change_role(_request(), 2, RoleUpdateForm(role=UserRole.MANAGER), ADMIN, cast(Any, users))
    await toggle_user(_request(), 2, ADMIN, cast(Any, users))
    # Re-saving their own (unchanged) admin role is harmless and allowed.
    await change_role(_request(), ADMIN.id, RoleUpdateForm(role=UserRole.ADMIN), ADMIN, cast(Any, users))
    assert users.role_changes == [(2, UserRole.MANAGER), (1, UserRole.ADMIN)]
    assert users.toggled == [2]
