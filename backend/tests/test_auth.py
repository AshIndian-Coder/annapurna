from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_login_success() -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "kitchen@example.com",
            "password": "demo123",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["token_type"] == "bearer"
    assert data["user"]["role"] == "KITCHEN"
    assert data["access_token"]


def test_login_invalid_password() -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "kitchen@example.com",
            "password": "wrong",
        },
    )

    assert response.status_code == 401


def test_me_requires_authentication() -> None:
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_me_with_valid_token() -> None:
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "kitchen@example.com",
            "password": "demo123",
        },
    )

    token = login_response.json()["access_token"]

    response = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["role"] == "KITCHEN"
