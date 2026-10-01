from uuid import UUID

from app.schemas.auth import LoginResponse, LoginRequest, UserResponse
from app.utils.auth import create_access_token


DEMO_USER = {
    "id": UUID("00000000-0000-0000-0000-000000000001"),
    "name": "Demo Kitchen",
    "email": "kitchen@example.com",
    "password": "demo123",
    "role": "KITCHEN",
}


def login(request: LoginRequest) -> LoginResponse:
    if (
        request.email.lower() != DEMO_USER["email"]
        or request.password != DEMO_USER["password"]
    ):
        raise ValueError("Invalid email or password")

    user = UserResponse(
        id=DEMO_USER["id"],
        name=DEMO_USER["name"],
        email=DEMO_USER["email"],
        role=DEMO_USER["role"],
    )

    token = create_access_token(
        subject=str(user.id),
        role=user.role,
    )

    return LoginResponse(
        access_token=token,
        user=user,
    )
