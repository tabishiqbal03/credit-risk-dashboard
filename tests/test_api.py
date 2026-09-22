from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["final_credit_decision_by_llm"] is False


def test_sensitive_tool_requires_approval():
    response = client.post(
        "/tools/request-evidence",
        json={"case_id": "CR-002", "missing_items": ["MISSING_INCOME_EVIDENCE"], "approved": False},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
