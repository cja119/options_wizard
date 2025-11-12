"""
Logistic regression model implementation.
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from .base import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = None

    @staticmethod
    def serialise(**kwargs) -> None:
        tick = kwargs["tick"]
        tmp_dir = Path(kwargs.get("tmp_dir", "./tmp/json/"))
        params = kwargs["model_params"]

        payload = {
            "coef_": params["coef_"],
            "intercept_": params["intercept_"],
            "classes_": params["classes_"],
            "hyperparams": params["hyperparams"],
            "X_cols": params["X_cols"],
            "fixed_effect_columns": params["fixed_effect_columns"],
            "fixed_effect_col": params.get("fixed_effect_col"),
        }
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / f"{tick}_logreg.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load(tick: str, tmp_dir: str = "./tmp/json/") -> dict:
        return json.loads(Path(tmp_dir, f"{tick}_logreg.json").read_text())

    @staticmethod
    def fit(outputs, X_cols: list[str], y_col: str, **kwargs):
        fixed_effect_col = kwargs.get("diff_ticks")
        threshold = kwargs.get("threshold", 0.0)
        drop_outliers = kwargs.get("drop_outliers", True)

        y = outputs[y_col].apply(lambda x: 1 if x >= threshold else 0).to_numpy()
        X_df = outputs[X_cols].copy()

        if fixed_effect_col and fixed_effect_col in outputs.columns:
            dummies = pd.get_dummies(outputs[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            X_df = pd.concat([X_df, dummies], axis=1)
            fixed_effect_columns = dummies.columns.tolist()
        else:
            fixed_effect_columns = []

        if drop_outliers:
            z = np.abs((X_df - X_df.mean()) / X_df.std(ddof=0))
            mask = (z < 3).all(axis=1)
            X_df = X_df[mask]
            y = y[mask.to_numpy()]

        model = SklearnLogisticRegression(max_iter=kwargs.get("max_iter", 1000))
        model.fit(X_df.values, y)

        params = {
            "coef_": model.coef_.tolist(),
            "intercept_": model.intercept_.tolist(),
            "classes_": model.classes_.tolist(),
            "hyperparams": model.get_params(),
            "X_cols": X_df.columns.tolist(),
            "fixed_effect_columns": fixed_effect_columns,
            "fixed_effect_col": fixed_effect_col,
        }

        name = kwargs.get("name", "logreg_pred")
        empty_preds = pd.DataFrame(index=outputs.index, columns=[name])
        return name, params, empty_preds

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs):
        params = kwargs.get("model_params") or LogisticRegression.load(
            kwargs["tick"], kwargs.get("tmp_dir", "./tmp/json/")
        )
        fixed_effect_col = kwargs.get("diff_ticks") or params.get("fixed_effect_col")

        X_pred = X.copy()
        if fixed_effect_col and fixed_effect_col in X_pred.columns:
            dummies = pd.get_dummies(X_pred[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            dummies = dummies.reindex(columns=params.get("fixed_effect_columns", []), fill_value=0)
            X_pred = pd.concat([X_pred.drop(columns=[fixed_effect_col], errors="ignore"), dummies], axis=1)

        X_pred = X_pred.reindex(columns=params["X_cols"], fill_value=0)

        model = SklearnLogisticRegression(**params["hyperparams"])
        model.classes_ = np.array(params["classes_"])
        model.coef_ = np.array(params["coef_"])
        model.intercept_ = np.array(params["intercept_"])

        preds = pd.Series(model.predict_proba(X_pred.values)[:, 1], index=X.index, name=kwargs.get("name", "logreg_pred"))
        return preds.name, params, preds.to_frame(preds.name)

    @staticmethod
    def fit_predict(outputs, **kwargs):
        name, params, _ = LogisticRegression.fit(outputs, **kwargs)
        _, _, preds = LogisticRegression.predict(outputs, model_params=params, **kwargs)

        if kwargs.get("save_params"):
            LogisticRegression.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )

        return name, params, preds
