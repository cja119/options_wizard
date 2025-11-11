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
        """Fit a small neural network regression model, optionally with fixed effects."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import pandas as pd
        import numpy as np

        fixed_effect_col = kwargs.get("fixed_effect_col") 

        X_df = outputs[X_cols].copy()
        y = outputs[y_col].values

        if fixed_effect_col is not None and fixed_effect_col in outputs.columns:
            dummies = pd.get_dummies(outputs[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            X_df = pd.concat([X_df, dummies], axis=1)

        X = X_df.values

        hidden_layer_sizes = kwargs.get("hidden_layer_sizes")
        if hidden_layer_sizes is None:
            best_size, hidden_layer_sizes = MultiLayerPerceptron.select_hidden_layer_size(
                outputs,
                X_cols,
                y_col,
                activation = activation,
                max_iter = kwargs.get("max_iter", 10000),
                lr = kwargs.get("learning_rate_init", 0.001),
                alpha=kwargs.get("alpha", 0.001),
                fixed_effect_col=kwargs.get("fixed_effect_col"),
            )
        activation = kwargs.get("activation", "relu")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver="adam",
                alpha=kwargs.get("alpha", 0.001),
                learning_rate_init=kwargs.get("learning_rate_init", 0.001),
                early_stopping=True,
                max_iter=kwargs.get("max_iter", 10000),
                random_state=kwargs.get("random_state", 42),
            ))
        ])
        model.fit(X, y)

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
            "X_cols": X_df.columns.tolist()  # save full column list for prediction
        }

        return params

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs) -> pd.Series:
        params = kwargs.get("model_params") or MultiLayerPerceptron.load(
            kwargs["tick"],
            kwargs.get("tmp_dir", "./tmp/json/")
        )

        X_pred = X.copy()

        # Recreate tick fixed-effect dummies if a column was used
        fixed_effect_col = kwargs.get("fixed_effect_col")
        if fixed_effect_col and fixed_effect_col in X_pred.columns:
            dummies = pd.get_dummies(
                X_pred[fixed_effect_col],
                prefix=fixed_effect_col,
                drop_first=True
            )
            # ensure we have exactly the columns the model was trained on
            fe_cols = params.get("fixed_effect_columns", [])
            dummies = dummies.reindex(columns=fe_cols, fill_value=0)
            X_pred = pd.concat([X_pred, dummies], axis=1)

        # align final column order (fill missing numeric cols with 0)
        X_pred = X_pred.reindex(columns=params["X_cols"], fill_value=0)

        mean = np.array(params["scaler_mean_"])
        scale = np.array(params["scaler_scale_"])
        scale = np.where(scale == 0, 1.0, scale)  # guard constant cols
        X_scaled = (X_pred.values - mean) / scale

        weights = [np.array(w) for w in params["coefs_"]]
        biases = [np.array(b) for b in params["intercepts_"]]

        def relu(x): return np.maximum(0, x)
        def logistic(x): return 1 / (1 + np.exp(-x))
        def identity(x): return x

        activations = {
            "relu": relu,
            "tanh": np.tanh,
            "logistic": logistic,
            "identity": identity,
        }
        activation_fn = activations.get(params["activation"], identity)

        a = X_scaled
        for W, b in zip(weights[:-1], biases[:-1]):
            a = activation_fn(a @ W + b)
        preds = a @ weights[-1] + biases[-1]
        return pd.Series(preds.flatten(), index=X.index, name=kwargs.get("name", "mlp_pred"))

    @staticmethod
    def fit_predict(outputs, **kwargs):
        name = kwargs.get("name", "mlp_pred")
        X_cols = kwargs.get("X_cols", [])
        fixed_effect_col = kwargs.get("diff_ticks", None)

        # Fit model
        params = MultiLayerPerceptron.fit(outputs, **kwargs, fixed_effect_col=fixed_effect_col)

        # Prepare X for prediction, include fixed effects
        X_pred = outputs[X_cols].copy()
         
        if fixed_effect_col is not None and fixed_effect_col in outputs.columns:
            dummies = pd.get_dummies(outputs[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            # align to training columns
            extra_cols = params["X_cols"][len(X_cols):]
            X_pred = pd.concat([X_pred, dummies.reindex(columns=extra_cols, fill_value=0)], axis=1)

        predictions = MultiLayerPerceptron.predict(X_pred, model_params=params)
        predictions = pd.DataFrame(predictions, index=outputs.index, columns=[name])

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
    
    @staticmethod
    def load(tick: str, tmp_dir: str = "./tmp/json/") -> dict:
        return json.loads(Path(tmp_dir, f"{tick}_mlpreg.json").read_text())

    @staticmethod
    def select_hidden_layer_size(
        outputs: pd.DataFrame,
        X_cols: list[str],
        y_col: str,
        activation: str,
        max_iter: int,
        lr: float,
        alpha: float,
        *,
        fixed_effect_col: str | None = None,
        candidates: tuple[tuple[int,...], ...] = ((8,), (16,), (32,), (64,), (8, 8), (16, 16)),
        k_folds: int = 5,
        random_state: int = 42,
    ) -> tuple[int, tuple[int, ...]]:
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.neural_network import MLPRegressor

        if not candidates:
            raise ValueError("Provide at least one candidate hidden-layer size.")

        X = outputs[X_cols].copy()
        if fixed_effect_col and fixed_effect_col in outputs.columns:
            dummies = pd.get_dummies(outputs[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

        y = outputs[y_col].values
        X_np = X.values

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        best_size, best_score = candidates[0], float("-inf")

        for size in candidates:
            fold_scores = []
            for train_idx, val_idx in kf.split(X_np):
                X_train, X_val = X_np[train_idx], X_np[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=size,
                        activation=activation,
                        solver="adam",
                        alpha=alpha,
                        learning_rate_init=lr,
                        early_stopping=True,
                        max_iter=max_iter,
                        random_state=random_state,
                    )),
                ])
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                fold_scores.append(r2_score(y_val, preds))

            score = float(np.mean(fold_scores))
            if score > best_score:
                best_score = score
                best_size = size

        return best_size, (best_size,)