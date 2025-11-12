"""
Linear regression model implementation.
"""

from __future__ import annotations
from pathlib import Path
import json

import pandas as pd
import numpy as np

from .base import BaseModel



class LinearRegression(BaseModel):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = None
        return None
    
    @staticmethod
    def serialise(**kwargs) -> None:
        tick = kwargs["tick"]
        tmp_dir = Path(kwargs.get("tmp_dir", "./tmp/json/"))
        params = kwargs["model_params"]

        payload = {
            "coef_": params["coef_"],
            "intercept_": params["intercept_"],
            "X_cols": params["X_cols"],
            "fixed_effect_columns": params["fixed_effect_columns"],
        }
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / f"{tick}_linreg.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load(tick: str, tmp_dir: str = "./tmp/json/") -> dict:
        return json.loads(Path(tmp_dir, f"{tick}_linreg.json").read_text())


    @staticmethod
    def fit(outputs, X_cols: list[str], y_col: str, **kwargs) -> dict:
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression

        fixed_effect_col = kwargs.get("diff_ticks", None)
        X_df = outputs[X_cols].copy()
        y = outputs[y_col]

        if fixed_effect_col and fixed_effect_col in outputs.columns:
            dummies = pd.get_dummies(outputs[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            X_df = pd.concat([X_df, dummies], axis=1)
            fixed_effect_columns = dummies.columns.tolist()
        else:
            fixed_effect_columns = []

        model = SklearnLinearRegression().fit(X_df, y)

        params = {
            "coef_": model.coef_.tolist(),
            "intercept_": float(model.intercept_),
            "X_cols": X_df.columns.tolist(),            # full feature order
            "fixed_effect_columns": fixed_effect_columns,
        }
        name = kwargs.get("name", "linreg_pred")
        empty_preds = pd.DataFrame(index=outputs.index, columns=[name])
        return name, params, empty_preds
        
    @staticmethod
    def predict(X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        params = kwargs.get("model_params") or LinearRegression.load(
            kwargs["tick"], kwargs.get("tmp_dir", "./tmp/json/")
        )
        fixed_effect_col = kwargs.get("diff_ticks", None)

        X_pred = X.copy()
        if fixed_effect_col and fixed_effect_col in X_pred.columns:
            dummies = pd.get_dummies(X_pred[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            dummies = dummies.reindex(columns=params.get("fixed_effect_columns", []), fill_value=0)
            X_pred = pd.concat([X_pred, dummies], axis=1)

        X_pred = X_pred.reindex(columns=params["X_cols"], fill_value=0)

        coef = np.array(params["coef_"])
        intercept = float(params["intercept_"])
        name = kwargs.get("name", "linreg_pred")
        series = pd.Series(X_pred.to_numpy() @ coef + intercept, index=X.index, name=name)
        return name, params, series.to_frame(name)

    
    @staticmethod
    def fit_predict(outputs, **kwargs):
        name = kwargs.get("name", "linreg_pred")
        _, params, _ = LinearRegression.fit(outputs, **kwargs)
        _, _, preds = LinearRegression.predict(outputs, model_params=params, **kwargs)

        if kwargs.get("save_params"):
            LinearRegression.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )
        return name, params, preds
