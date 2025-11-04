"""
Linear regression model implementation.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import json

import pandas as pd

from src.model.base import BaseModel



class LinearRegression(BaseModel):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = None
        return None
    
    @staticmethod
    def serialise(**kwargs) -> None:
        tick = kwargs.get("tick", "")
        tmp_dir = kwargs.get("tmp_dir", "./tmp/json/")
        model_params = kwargs.get("model_params", None)

        if model_params is None:
            raise ValueError("model_params must be provided for serialization.")

        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serialisable = {
            "coef_": model_params["coef_"],
            "intercept_": model_params["intercept_"],
        }
        with open(f"{tmp_dir}/{tick}_linreg.json", "w") as f:
            json.dump(serialisable, f, indent=2)

    @staticmethod
    def fit(outputs, X_cols: list[str], y_col: str) -> dict:
        """Fit linear regression model."""
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression

        X = outputs[X_cols]
        y = outputs[y_col]

        model = SklearnLinearRegression()
        model.fit(X, y)

        # Return learned parameters for serialization
        params = {
            "coef_": model.coef_.tolist(),
            "intercept_": model.intercept_.tolist()
            if hasattr(model.intercept_, "__len__")
            else model.intercept_,
        }

        return params

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs) -> pd.Series:
        """Predict using linear regression model parameters."""
        params = kwargs.get("model_params")
        if params is None:
            raise ValueError("Model parameters must be provided for prediction.")

        import numpy as np

        coef = np.array(params["coef_"])
        intercept = params["intercept_"]

        preds = X.values @ coef + intercept
        return pd.Series(preds, index=X.index)

    @staticmethod
    def fit_predict(outputs, **kwargs):
        """Fit model, generate predictions, and return pipeline-compatible tuple."""
        name = kwargs.get("name", "linear_regression_prediction")
        X_cols = kwargs.get("X_cols", [])
        y_col = kwargs.get("y_col", "")

        # Fit model
        params = LinearRegression.fit(outputs, X_cols, y_col)

        # Predict
        predictions = LinearRegression.predict(outputs[X_cols], model_params=params)

        # Convert to DataFrame for compatibility
        predictions = pd.DataFrame(predictions, index=outputs.index, columns=[name])

        # Optionally save parameters (e.g., as JSON)
        if kwargs.get("save_params", False):
            LinearRegression.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )

        return name, params, predictions
