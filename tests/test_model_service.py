from src.credit.model_service import CreditModelService


def test_missing_model_artifacts_degrade_gracefully(tmp_path):
    service = CreditModelService(tmp_path / "models")
    prediction = service.predict_application({"declared_annual_income": 50000})
    assert prediction.available is False
    assert prediction.decision_band == "unavailable"
    assert "python train.py" in prediction.note
