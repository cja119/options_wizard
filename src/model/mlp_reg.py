"""
Neural network model implementation (MLPRegressor-based) with target-encoded fixed effects.
"""

from __future__ import annotations
from pathlib import Path
import base64
import json
import pickle

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .base import BaseModel


class MultiLayerPerceptron(BaseModel):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _build_pipeline(hidden_layer_sizes, activation, alpha, lr, max_iter, random_state):
        return Pipeline([
            ("encoder", TargetEncoder(min_samples_leaf = 5, )),
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver="adam",
                alpha=alpha,
                learning_rate_init=lr,
                early_stopping=False,
                max_iter=max_iter,
                random_state=random_state,
            )),
        ])

    @staticmethod
    def _dump_encoder(encoder: TargetEncoder) -> str:
        """Serialize the fitted encoder to a base64 string."""
        blob = pickle.dumps(encoder)
        return base64.b64encode(blob).decode("utf-8")

    @staticmethod
    def _load_encoder(payload: str) -> TargetEncoder:
        """Rehydrate a TargetEncoder from the stored base64 string."""
        blob = base64.b64decode(payload.encode("utf-8"))
        return pickle.loads(blob)

    @staticmethod
    def serialise(**kwargs) -> None:
        tick = kwargs["tick"]
        tmp_dir = Path(kwargs.get("tmp_dir", "./tmp/json/"))
        params = kwargs["model_params"]

        payload = {
            "encoder": params["encoder"],
            "scaler_mean_": params["scaler_mean_"],
            "scaler_scale_": params["scaler_scale_"],
            "coefs_": params["coefs_"],
            "intercepts_": params["intercepts_"],
            "hidden_layer_sizes": params["hidden_layer_sizes"],
            "activation": params["activation"],
            "alpha": params["alpha"],
            "learning_rate_init": params["learning_rate_init"],
            "X_cols": params["X_cols"],
            "fixed_effect_col": params.get("fixed_effect_col"),
        }
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / f"{tick}_mlpreg.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load(tick: str, tmp_dir: str = "./tmp/json/") -> dict:
        return json.loads(Path(tmp_dir, f"{tick}_mlpreg.json").read_text())

    # ------------------------------------------------------------------ fit / predict
    @staticmethod
    def fit(outputs, X_cols: list[str], y_col: str, **kwargs):
        hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (16,))
        activation = kwargs.get("activation", "relu")
        alpha = kwargs.get("alpha", 0.001)
        lr = kwargs.get("learning_rate_init", 0.001)
        max_iter = kwargs.get("max_iter", 1000)
        random_state = kwargs.get("random_state", 42)
        name = kwargs.get("name", "mlp_pred")

        fixed_effect_col = kwargs.get("diff_ticks")
        X_df = outputs[X_cols].copy()
        y = outputs[y_col].values

        if fixed_effect_col and fixed_effect_col in outputs.columns:
            X_df = pd.concat([X_df, outputs[[fixed_effect_col]]], axis=1)
            encoder_cols = [fixed_effect_col]
        else:
            encoder_cols = []
        
        if kwargs.get('x_val', True):
            hidden_layer_grid = kwargs.get("hidden_layer_grid", [(8,), (16,), (4,4), (8, 8), (32,)])
            wf_kwargs = kwargs.get(
                "walk_forward_kwargs", {
                "min_train_size": 50,
                "val_window": 20,
                "metric": "mse",
                "fit_kwargs": {"alpha": 1e-4, "learning_rate_init": 1e-3},
                })

            if hidden_layer_grid:
                best_cfg = MultiLayerPerceptron._walk_forward_hidden_size(
                    outputs, X_cols, y_col,
                    hidden_layer_grid,
                    diff_ticks=fixed_effect_col,
                    **wf_kwargs,
                )
                hidden_layer_sizes = best_cfg['best_size']

        pipeline = MultiLayerPerceptron._build_pipeline(hidden_layer_sizes, activation, alpha, lr, max_iter, random_state)
        pipeline.set_params(encoder__cols=encoder_cols)
        pipeline.fit(X_df, y)

        encoder: TargetEncoder = pipeline.named_steps["encoder"]
        scaler: StandardScaler = pipeline.named_steps["scaler"]
        mlp: MLPRegressor = pipeline.named_steps["mlp"]

        params = {
            "encoder": MultiLayerPerceptron._dump_encoder(encoder),
            "scaler_mean_": scaler.mean_.tolist(),
            "scaler_scale_": scaler.scale_.tolist(),
            "coefs_": [coef.tolist() for coef in mlp.coefs_],
            "intercepts_": [inter.tolist() for inter in mlp.intercepts_],
            "hidden_layer_sizes": mlp.hidden_layer_sizes,
            "activation": mlp.activation,
            "alpha": mlp.alpha,
            "learning_rate_init": mlp.learning_rate_init,
            "X_cols": X_df.columns.tolist(),
            "fixed_effect_col": fixed_effect_col,
        }
        empty_preds = pd.DataFrame(index=outputs.index, columns=[name])
        return name, params, empty_preds

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs):
        params = kwargs.get("model_params") or MultiLayerPerceptron.load(
            kwargs["tick"], kwargs.get("tmp_dir", "./tmp/json/")
        )
        fixed_effect_col = kwargs.get("diff_ticks") or params.get("fixed_effect_col")
        name = kwargs.get("name", "mlp_pred")
        if fixed_effect_col:
            if fixed_effect_col not in X.columns:
                raise ValueError(f"Missing '{fixed_effect_col}' column required for target encoding")

        X_pred = X.copy()
        if fixed_effect_col and fixed_effect_col in X_pred.columns:
            X_pred = X_pred.drop(columns=[fixed_effect_col])
            X_pred[fixed_effect_col] = X[fixed_effect_col]

        X_pred = X_pred.reindex(columns=params["X_cols"], fill_value=0)

        encoder = MultiLayerPerceptron._load_encoder(params["encoder"])
        X_encoded = encoder.transform(X_pred.copy())

        mean = np.array(params["scaler_mean_"])
        scale = np.array(params["scaler_scale_"])
        scale = np.where(scale == 0, 1.0, scale)
        X_scaled = (X_encoded.values - mean) / scale

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
        series = pd.Series(preds.flatten(), index=X.index, name=name)
        return name, params, series.to_frame(name)

    @staticmethod
    def fit_predict(outputs, **kwargs):
        name, params, _ = MultiLayerPerceptron.fit(outputs, **kwargs)
        _, _, preds = MultiLayerPerceptron.predict(outputs, model_params=params, **kwargs)

        if kwargs.get("save_params"):
            MultiLayerPerceptron.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )
        return name, params, preds
    
    @staticmethod
    def _walk_forward_hidden_size(
        outputs: pd.DataFrame,
        X_cols: list[str],
        y_col: str,
        hidden_layer_grid: list[tuple[int, ...]],
        *,
        diff_ticks: str | None = None,
        min_train_size: int = 252,
        val_window: int = 21,
        step_size: int | None = None,
        metric: str = "mse",
        **fit_kwargs,
    ) -> dict:
        """
        Walk-forward validation for hidden-layer selection.
        """
        row_count = len(outputs)
        if step_size is None:
            step_size = max(1, int((row_count - min_train_size) * (val_window / 100)))


        data = outputs.sort_index()
        scorer = mean_squared_error if metric == "mse" else \
                 (lambda a, b: np.mean(np.abs(a - b)))

        per_split_scores = {size: [] for size in hidden_layer_grid}

        split_start = min_train_size
        while split_start + val_window <= len(data):
            train = data.iloc[:split_start]
            val = data.iloc[split_start:split_start + val_window]

            for size in hidden_layer_grid:
                local_kwargs = {
                    "hidden_layer_sizes": size,
                    "diff_ticks": diff_ticks,
                    'x_val': False,
                    **fit_kwargs,
                }
                local_kwargs.pop("hidden_layer_grid", None)
                _, params, _ = MultiLayerPerceptron.fit(train, X_cols, y_col, **local_kwargs)
                _, _, preds = MultiLayerPerceptron.predict(
                    val,
                    model_params=params,
                    diff_ticks=diff_ticks,
                    name="mlp_pred",
                )
                score = scorer(val[y_col].values, preds["mlp_pred"].values)
                per_split_scores[size].append(score)

            split_start += step_size

        avg_scores = {
            size: np.mean(scores) for size, scores in per_split_scores.items() if scores
        }
        if not avg_scores:
            raise ValueError("No validation folds were produced. Adjust window/step sizes.")

        best_size = min(avg_scores, key=avg_scores.get)
        return {
            "best_size": best_size,
            "avg_scores": avg_scores,
            "per_split_scores": per_split_scores,
        }
