"""
Logistic regression model implementation.
"""

from __future__ import annotations
from pathlib import Path
import json

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import numpy as np
import pandas as pd

from .base import BaseModel



class LogisticRegression(BaseModel):
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
            "coef_": model_params["coef_"].tolist(),
            "intercept_": model_params["intercept_"].tolist(),
            "classes_": model_params["classes_"].tolist(),
            "hyperparams": model_params["hyperparams"],
        }

        with open(f"{tmp_dir}/{tick}_logreg.json", "w") as f:
            json.dump(serialisable, f, indent=2)
    
    @staticmethod
    def fit_predict(outputs, **kwargs) -> None:
        "Fit and predict using logistic regression model."
        name = kwargs.get("name", "logistic_regression_prediction")
        X_cols = kwargs.get("X_cols", [])
        y_col = kwargs.get("y_col", "")
        params = LogisticRegression.fit(outputs, X_cols, y_col)
        predictions = LogisticRegression.predict(outputs[X_cols], model_params=params)
        if kwargs.get("save_params", False):
            LogisticRegression.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )
        predictions =  pd.DataFrame(pd.Series(predictions, index=outputs.index, name=name))
        return name, params, predictions
    
    @staticmethod
    def fit(outputs, X_cols: list[str], y_col: str, threshold: float = 0.0, drop_outliers: bool = True):
        """Fit logistic regression model, optionally dropping outliers."""
        
        y = outputs[y_col].apply(lambda x: 1 if x >= threshold else 0)
        X = outputs[X_cols].copy()
        
        if drop_outliers:
            z_scores = np.abs((X - X.mean()) / X.std(ddof=0))
            mask = (z_scores < 3).all(axis=1)
            X = X[mask]
            y = y[mask]

        model = SklearnLogisticRegression(max_iter=1000)
        model.fit(X, y)

        params = {
            "coef_": model.coef_,
            "intercept_": model.intercept_,
            "classes_": model.classes_,
            "hyperparams": model.get_params(),
        }

        return params
    
        
    @staticmethod
    def load(tick: str, tmp_dir: str = "./tmp/json/"):
        """Load logistic regression parameters from JSON and rebuild model."""
        path = Path(tmp_dir) / f"{tick}_logreg.json"
        if not path.exists():
            raise FileNotFoundError(f"No saved model found for {tick} at {path}")

        with open(path, "r") as f:
            params = json.load(f)

        # Convert back to numpy arrays
        model = SklearnLogisticRegression(**params["hyperparams"])
        model.classes_ = np.array(params["classes_"])
        model.coef_ = np.array(params["coef_"])
        model.intercept_ = np.array(params["intercept_"])

        return model
    
    @staticmethod
    def predict(X, **kwargs):
        """Predict using logistic regression from saved parameters."""
        from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
        params = kwargs.get("model_params", None)
        if params is None:
            params = LogisticRegression.load(kwargs.get("tick", ""))
        
        model = SklearnLogisticRegression(**params["hyperparams"])
        model.classes_ = params["classes_"]
        model.coef_ = params["coef_"]
        model.intercept_ = params["intercept_"]

        return model.predict(X)
