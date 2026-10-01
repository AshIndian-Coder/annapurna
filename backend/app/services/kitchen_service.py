from datetime import date, datetime, timezone
from uuid import UUID

from app.schemas.kitchen import (
    AttendanceCreate,
    AttendanceResponse,
    MenuItem,
    MenuResponse,
)


def record_attendance(
    kitchen_id: UUID,
    payload: AttendanceCreate,
) -> AttendanceResponse:
    """
    Temporary service implementation.

    Persistence will be connected once the database schema
    and repository contract are available.
    """
    return AttendanceResponse(
        kitchen_id=kitchen_id,
        date=date.today(),
        staff_count=payload.staff_count,
        recorded_at=datetime.now(timezone.utc),
    )


def get_menu(kitchen_id: UUID) -> MenuResponse:
    """
    Temporary service implementation.

    Returns an empty menu until the database integration
    is connected.
    """
    return MenuResponse(
        kitchen_id=kitchen_id,
        date=date.today(),
        items=[],
    )
