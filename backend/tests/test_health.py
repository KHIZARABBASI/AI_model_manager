from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ai-model-manager-backend",
    }


def test_hardware_check() -> None:
    response = client.get("/hardware")

    assert response.status_code == 200
    payload = response.json()
    assert payload["os_name"]
    assert payload["cpu_name"]
    assert payload["logical_cpus"] >= 1
    assert isinstance(payload["gpus"], list)
