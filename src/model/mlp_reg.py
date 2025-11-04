"""
Neural network model implementation (MLPRegressor-based).
"""

from __future__ import annotations
import json
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np
import json 

from src.model.base import BaseModel


class MultiLayerPerceptron(BaseModel):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = None
        return None

    @staticmethod
    def fit(outputs, X_cols: list[str], y_col: str, **kwargs) -> dict:
        """Fit a small neural network regression model."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X = outputs[X_cols].values
        y = outputs[y_col].values

        # Allow configurable architecture
        hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (16,))
        activation = kwargs.get("activation", "relu")

        # Define a modest, regularized MLP
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver="adam",
                alpha=kwargs.get("alpha", 0.001),
                learning_rate_init=kwargs.get("learning_rate_init", 0.001),
                early_stopping=True,
                max_iter=kwargs.get("max_iter", 1000),
                random_state=kwargs.get("random_state", 42),
            ))
        ])

        model.fit(X, y)

        # Extract learned parameters for serialization
        mlp = model.named_steps["mlp"]
        scaler = model.named_steps["scaler"]

        params = {
            "coefs_": [coef.tolist() for coef in mlp.coefs_],
            "intercepts_": [inter.tolist() for inter in mlp.intercepts_],
            "hidden_layer_sizes": mlp.hidden_layer_sizes,
            "activation": mlp.activation,
            "alpha": mlp.alpha,
            "learning_rate_init": mlp.learning_rate_init,
            "scaler_mean_": scaler.mean_.tolist(),
            "scaler_scale_": scaler.scale_.tolist(),
        }

        return params

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs) -> pd.Series:
        """Predict using trained neural network parameters."""
        from sklearn.neural_network import MLPRegressor
        import numpy as np

        params = kwargs.get("model_params")
        if params is None:
            raise ValueError("Model parameters must be provided for prediction.")

        # Reconstruct scaler
        X_scaled = (X - np.array(params["scaler_mean_"])) / np.array(params["scaler_scale_"])

        # Rebuild MLP manually
        weights = [np.array(w) for w in params["coefs_"]]
        biases = [np.array(b) for b in params["intercepts_"]]

        def relu(x): return np.maximum(0, x)
        def identity(x): return x

        activation_fn = relu if params["activation"] == "relu" else identity

        # Forward pass manually
        a = X_scaled
        for i in range(len(weights) - 1):
            a = activation_fn(a @ weights[i] + biases[i])
        preds = a @ weights[-1] + biases[-1]

        return pd.DataFrame(preds, index=X.index)

    @staticmethod
    def fit_predict(outputs, **kwargs):
        """Fit model, generate predictions, and return pipeline-compatible tuple."""
        name = kwargs.get("name", "multi_layer_perceptron_prediction")
        X_cols = kwargs.get("X_cols", [])
        y_col = kwargs.get("y_col", "")

        # Fit model
        params = MultiLayerPerceptron.fit(outputs, X_cols, y_col)

        # Predict
        predictions = MultiLayerPerceptron.predict(outputs[X_cols], model_params=params)

        # Convert to DataFrame for pipeline compatibility
        predictions = pd.DataFrame(predictions, index=outputs.index, columns=[name])

        # Optionally save parameters
        if kwargs.get("save_params", False):
            MultiLayerPerceptron.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )

        return name, params, predictions

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
            "coefs_": model_params["coefs_"],
            "intercepts_": model_params["intercepts_"],
            "hidden_layer_sizes": model_params["hidden_layer_sizes"],
            "activation": model_params["activation"],
            "alpha": model_params["alpha"],
            "learning_rate_init": model_params["learning_rate_init"],
            "scaler_mean_": model_params["scaler_mean_"],
            "scaler_scale_": model_params["scaler_scale_"],
        }
        with open(f"{tmp_dir}/{tick}_mlpreg.json", "w") as f:
            json.dump(serialisable, f, indent=2)