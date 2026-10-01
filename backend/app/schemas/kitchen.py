from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class AttendanceCreate(BaseModel):
    staff_count: int = Field(..., ge=0)


class AttendanceResponse(BaseModel):
    kitchen_id: UUID
    date: date
    staff_count: int
    recorded_at: datetime


class MenuItem(BaseModel):
    name: str
    quantity: float = Field(..., ge=0)
    unit: str = "kg"


class MenuResponse(BaseModel):
    kitchen_id: UUID
    date: date
    items: list[MenuItem]
