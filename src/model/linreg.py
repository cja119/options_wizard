"""
Linear regression model implementation.
"""

from __future__ import annotations
from typing import Optional
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
        """Fit linear regression model, optionally with fixed effects."""
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression
        import pandas as pd

        fixed_effect_col = kwargs.get("fixed_effect_col", None)

        X = outputs[X_cols].copy()
        y = outputs[y_col]
        
        ## Fit method, after creating dummy columns
        if fixed_effect_col is not None and fixed_effect_col in outputs.columns:
            dummies = pd.get_dummies(outputs[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            fixed_effect_columns = dummies.columns.tolist()  # store column names
        else:
            fixed_effect_columns = []

        model = SklearnLinearRegression()
        model.fit(X, y)

        params = {
            "coef_": model.coef_.tolist(),
            "intercept_": model.intercept_.tolist(),
            "X_cols": X_cols,  # original feature names
            "fixed_effect_columns": fixed_effect_columns  # new!
        }
        return params
    
    @staticmethod
    def predict(X: pd.DataFrame, **kwargs) -> pd.Series:
        params = kwargs.get("model_params") or LinearRegression.load(kwargs["tick"], kwargs.get("tmp_dir"))
        X_pred = X.reindex(columns=params["X_cols"], copy=True)

        fe_col = kwargs.get("fixed_effect_col")
        if fe_col:
            dummies = pd.get_dummies(X[fe_col], prefix=fe_col, drop_first=True)
            dummies = dummies.reindex(columns=params["fixed_effect_columns"], fill_value=0)
            X_pred = pd.concat([X_pred, dummies], axis=1)

        coef = np.array(params["coef_"])
        intercept = np.array(params["intercept_"])
        return pd.Series(X_pred.values @ coef + intercept, index=X.index)
    
    @staticmethod
    def fit_predict(outputs, **kwargs):
        """Fit model, generate predictions, and return pipeline-compatible tuple."""
        name = kwargs.get("name", "linear_regression_prediction")
        X_cols = kwargs.get("X_cols", [])
        y_col = kwargs.get("y_col", "")
        fixed_effect_col = kwargs.get("diff_ticks", None) 

        # Fit model
        params = LinearRegression.fit(outputs, X_cols, y_col, fixed_effect_col=fixed_effect_col)

        # Predict
        predictions = LinearRegression.predict(
            outputs,
            X_cols=X_cols,
            model_params=params,
            fixed_effect_col=fixed_effect_col
        )

        # Convert to DataFrame for pipeline compatibility
        predictions = pd.DataFrame(predictions, index=outputs.index, columns=[name])

        # Optionally save parameters
        if kwargs.get("save_params", False):
            LinearRegression.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )

        # Return a tuple: (name, model parameters, predictions)
        return name, params, predictions
