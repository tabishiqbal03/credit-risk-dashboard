from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CreditPrediction:
    available: bool
    predicted_default_probability: float | None
    decision_band: str
    key_drivers: list[str]
    model_name: str | None
    model_version: str | None
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


class CreditModelService:
    """Loads the existing LightGBM artefacts without changing the training pipeline."""

    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_cols: list[str] = []
        self.feature_defaults: dict[str, float] = {}
        self.thresholds: dict[str, float] = {}
        self.model_version = None
        self._load()

    def _load_json(self, filename: str, default: Any = None) -> Any:
        path = self.models_dir / filename
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _load(self) -> None:
        model_path = self.models_dir / "lightgbm.pkl"
        if model_path.exists():
            with model_path.open("rb") as f:
                self.model = pickle.load(f)
        self.feature_cols = self._load_json("feature_cols.json", []) or []
        self.feature_defaults = self._load_json("feature_defaults.json", {}) or {}
        self.thresholds = self._load_json("decision_thresholds.json", {}) or {}
        version = self._load_json("model_version.json", {}) or {}
        self.model_version = version.get("version")

    @property
    def available(self) -> bool:
        return bool(
            self.model is not None
            and self.feature_cols
            and self.feature_defaults
            and "approve_threshold" in self.thresholds
            and "reject_threshold" in self.thresholds
        )

    def missing_artifacts(self) -> list[str]:
        required = [
            "lightgbm.pkl",
            "feature_cols.json",
            "feature_defaults.json",
            "decision_thresholds.json",
        ]
        return [name for name in required if not (self.models_dir / name).exists()]

    def build_demo_vector(self, application: dict) -> pd.DataFrame:
        if not self.feature_defaults or not self.feature_cols:
            raise RuntimeError("Training artefacts with feature defaults are required")
        row = {feature: float(self.feature_defaults.get(feature, 0.0)) for feature in self.feature_cols}

        mapping = {
            "AMT_INCOME_TOTAL": application.get("declared_annual_income"),
            "AMT_CREDIT": application.get("requested_credit"),
            "AMT_ANNUITY": application.get("annuity"),
        }
        for feature, value in mapping.items():
            if feature in row and value is not None:
                row[feature] = float(value)

        duration = application.get("employment_duration_months")
        if "DAYS_EMPLOYED" in row and duration is not None:
            row["DAYS_EMPLOYED"] = -float(duration) * 30.4375
        age = application.get("age_years")
        if "DAYS_BIRTH" in row and age is not None:
            row["DAYS_BIRTH"] = -float(age) * 365.25

        if "CREDIT_INCOME_RATIO" in row:
            row["CREDIT_INCOME_RATIO"] = row.get("AMT_CREDIT", 0.0) / (row.get("AMT_INCOME_TOTAL", 0.0) + 1.0)
        if "ANNUITY_INCOME_RATIO" in row:
            row["ANNUITY_INCOME_RATIO"] = row.get("AMT_ANNUITY", 0.0) / (row.get("AMT_INCOME_TOTAL", 0.0) + 1.0)
        if "CREDIT_TERM" in row:
            row["CREDIT_TERM"] = row.get("AMT_CREDIT", 0.0) / (row.get("AMT_ANNUITY", 0.0) + 1.0)
        if "DAYS_EMPLOYED_RATIO" in row:
            row["DAYS_EMPLOYED_RATIO"] = abs(row.get("DAYS_EMPLOYED", 0.0)) / (abs(row.get("DAYS_BIRTH", 0.0)) + 1.0)

        return pd.DataFrame([[row[c] for c in self.feature_cols]], columns=self.feature_cols)

    def _decision_band(self, probability: float) -> str:
        approve = float(self.thresholds["approve_threshold"])
        reject = float(self.thresholds["reject_threshold"])
        if probability < approve:
            return "approve"
        if probability > reject:
            return "reject"
        return "review"

    def _drivers(self, frame: pd.DataFrame, top_k: int = 8) -> list[str]:
        try:
            booster = getattr(self.model, "booster_", None)
            if booster is None:
                return []
            contrib = booster.predict(frame, pred_contrib=True)
            values = np.asarray(contrib)[0][:-1]
            order = np.argsort(-np.abs(values))[:top_k]
            return [f"{self.feature_cols[i]} ({values[i]:+.4f})" for i in order]
        except Exception:
            return []

    def predict_application(self, application: dict) -> CreditPrediction:
        if not self.available:
            missing = ", ".join(self.missing_artifacts()) or "required model artefacts"
            return CreditPrediction(
                False,
                None,
                "unavailable",
                [],
                None,
                self.model_version,
                f"Model scoring unavailable. Run `python train.py` to generate: {missing}.",
            )
        frame = self.build_demo_vector(application)
        probability = float(self.model.predict_proba(frame)[0, 1])
        return CreditPrediction(
            True,
            probability,
            self._decision_band(probability),
            self._drivers(frame),
            "LightGBM",
            self.model_version or "legacy-lightgbm",
            "Synthetic demo profile scored using training-feature medians for unspecified fields; this is not an original Home Credit applicant.",
        )
