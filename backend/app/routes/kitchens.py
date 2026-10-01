from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.schemas.kitchen import (
    AttendanceCreate,
    AttendanceResponse,
    MenuResponse,
)
from app.services.kitchen_service import (
    get_menu,
    record_attendance,
)
from app.utils.auth import require_role


router = APIRouter(
    prefix="/kitchens",
    tags=["Kitchens"],
)


@router.post(
    "/{kitchen_id}/attendance",
    response_model=AttendanceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_attendance(
    kitchen_id: UUID,
    payload: AttendanceCreate,
    _current_user: dict = Depends(require_role("KITCHEN", "ADMIN")),
) -> AttendanceResponse:
    return record_attendance(kitchen_id, payload)


@router.get(
    "/{kitchen_id}/menu",
    response_model=MenuResponse,
)
def read_menu(
    kitchen_id: UUID,
    _current_user: dict = Depends(
        require_role("KITCHEN", "NGO", "LOGISTICS", "ADMIN")
    ),
) -> MenuResponse:
    return get_menu(kitchen_id)
