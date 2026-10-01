from uuid import UUID

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

KITCHEN_ID = UUID("11111111-1111-1111-1111-111111111111")


def get_kitchen_token() -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "kitchen@example.com",
            "password": "demo123",
        },
    )

    assert response.status_code == 200

    return response.json()["access_token"]


def test_attendance_requires_authentication() -> None:
    response = client.post(
        f"/api/v1/kitchens/{KITCHEN_ID}/attendance",
        json={"staff_count": 12},
    )

    assert response.status_code == 401


def test_attendance_success() -> None:
    token = get_kitchen_token()

    response = client.post(
        f"/api/v1/kitchens/{KITCHEN_ID}/attendance",
        headers={"Authorization": f"Bearer {token}"},
        json={"staff_count": 12},
    )

    assert response.status_code == 201

    data = response.json()

    assert data["kitchen_id"] == str(KITCHEN_ID)
    assert data["staff_count"] == 12
    assert "date" in data
    assert "recorded_at" in data


def test_attendance_rejects_negative_staff_count() -> None:
    token = get_kitchen_token()

    response = client.post(
        f"/api/v1/kitchens/{KITCHEN_ID}/attendance",
        headers={"Authorization": f"Bearer {token}"},
        json={"staff_count": -1},
    )

    assert response.status_code == 422


def test_menu_requires_authentication() -> None:
    response = client.get(
        f"/api/v1/kitchens/{KITCHEN_ID}/menu",
    )

    assert response.status_code == 401


def test_menu_success() -> None:
    token = get_kitchen_token()

    response = client.get(
        f"/api/v1/kitchens/{KITCHEN_ID}/menu",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["kitchen_id"] == str(KITCHEN_ID)
    assert "date" in data
    assert isinstance(data["items"], list)
