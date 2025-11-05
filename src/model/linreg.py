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
    def fit(outputs, X_cols: list[str], y_col: str, **kwargs) -> dict:
        """Fit linear regression model, optionally with fixed effects."""
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression
        import pandas as pd

        fixed_effect_col = kwargs.get("fixed_effect_col")  # get from kwargs

        X = outputs[X_cols].copy()
        y = outputs[y_col]

        # Add fixed effects if specified
        if fixed_effect_col is not None and fixed_effect_col in outputs.columns:
            # One-hot encode the fixed effect, drop first to avoid multicollinearity
            dummies = pd.get_dummies(outputs[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

        model = SklearnLinearRegression()
        model.fit(X, y)

        # Return learned parameters for serialization
        params = {
            "coef_": model.coef_.tolist(),
            "intercept_": model.intercept_.tolist()
            if hasattr(model.intercept_, "__len__")
            else model.intercept_,
            "X_cols": X.columns.tolist()  # save columns for prediction
        }

        return params

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs) -> pd.Series:
        """Predict using linear regression model parameters."""
        import numpy as np
        params = kwargs.get("model_params")
        if params is None:
            raise ValueError("Model parameters must be provided for prediction.")

        X_cols = kwargs.get("X_cols")  # Columns used for training
        fixed_effect_col = kwargs.get("fixed_effect_col")

        # Start with original features
        X_pred = X[X_cols].copy()

        # Add fixed effects if applicable
        if fixed_effect_col is not None:
            fe_cols = params.get("fixed_effect_columns")  # stored during fit
            X_fe = pd.get_dummies(X[fixed_effect_col], drop_first=False)

            # Ensure same order and missing columns are filled with 0
            for col in fe_cols:
                if col not in X_fe:
                    X_fe[col] = 0
            X_fe = X_fe[fe_cols]  # reorder to match training
            X_pred = pd.concat([X_pred, X_fe], axis=1)

        coef = np.array(params["coef_"])
        intercept = params["intercept_"]

        preds = X_pred.values @ coef + intercept
        return pd.Series(preds, index=X.index)
    
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
        predictions = LinearRegression.predict(outputs, X_cols=X_cols, model_params=params, fixed_effect_col=fixed_effect_col)

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
