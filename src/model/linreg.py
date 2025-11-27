"""
Linear regression model implementation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable
import json

import pandas as pd
import numpy as np

from .base import BaseModel


def all_stocks(method: Callable) -> Callable:
    """Decorator to flag model methods that should run once across all stocks."""
    target = method.__func__ if isinstance(method, staticmethod) else method
    setattr(target, "_all_stocks", True)
    return method


class LinearRegression(BaseModel):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = None
        return None

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _resolve_xy(kwargs: dict) -> tuple[list[str], str]:
        """Resolve feature/target aliases from kwargs."""
        X_cols = kwargs.get("X_cols") or kwargs.get("features")
        y_col = kwargs.get("y_col") or kwargs.get("target")

        if X_cols is None:
            raise ValueError("You must provide feature columns via 'X_cols' or 'features'.")
        if y_col is None:
            raise ValueError("You must provide a target column via 'y_col' or 'target'.")
        return list(X_cols), str(y_col)

    @staticmethod
    def _build_feature_matrix(
        outputs: pd.DataFrame,
        X_cols: list[str],
        *,
        fixed_effect_col: str | None,
        train_mask: pd.Series | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
        """Prepare feature matrix with optional fixed effects and imputation."""
        df = outputs.copy()

        missing_cols = [col for col in X_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing feature columns for regression: {missing_cols}")

        X_df = df[X_cols].copy()
        # Coerce any nested objects (e.g., Series) to NaN so pandas ops don't choke.
        if X_df.applymap(lambda v: not pd.api.types.is_scalar(v)).any().any():
            X_df = X_df.where(X_df.applymap(pd.api.types.is_scalar), np.nan)

        fixed_effect_columns: list[str] = []
        if fixed_effect_col and fixed_effect_col in df.columns:
            dummies = pd.get_dummies(df[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            X_df = pd.concat([X_df.drop(columns=[fixed_effect_col], errors="ignore"), dummies], axis=1)
            fixed_effect_columns = dummies.columns.tolist()

        # Ensure unique column names before downstream reindexing.
        X_df = X_df.loc[:, ~X_df.columns.duplicated()]

        # Force numeric; anything non-numeric becomes NaN and is handled by fill.
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        stats_mask = train_mask if train_mask is not None else pd.Series(True, index=X_df.index)
        fill_values = X_df[stats_mask].median(numeric_only=True)
        X_df = X_df.fillna(fill_values).fillna(0)

        return df, X_df, fill_values, fixed_effect_columns

    @staticmethod
    def serialise(**kwargs) -> None:
        tick = kwargs["tick"]
        tmp_dir = Path(kwargs.get("tmp_dir", "./tmp/json/"))
        params = kwargs["model_params"]

        payload = {
            "coef_": params["coef_"],
            "intercept_": params["intercept_"],
            "X_cols": params["X_cols"],
            "base_features": params.get("base_features", []),
            "fill_values": params.get("fill_values", {}),
            "fixed_effect_columns": params["fixed_effect_columns"],
            "fixed_effect_col": params.get("fixed_effect_col"),
        }
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / f"{tick}_linreg.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load(tick: str, tmp_dir: str = "./tmp/json/") -> dict:
        return json.loads(Path(tmp_dir, f"{tick}_linreg.json").read_text())

    @staticmethod
    def fit(outputs, **kwargs) -> dict:
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression

        X_cols, y_col = LinearRegression._resolve_xy(kwargs)
        fixed_effect_col = kwargs.get("diff_ticks")

        train_mask = None
        if kwargs.get("train_test_split") and "set" in outputs.columns:
            train_mask = outputs["set"].str.lower() == "train"

        df, X_df, fill_values, fixed_effect_columns = LinearRegression._build_feature_matrix(
            outputs,
            X_cols,
            fixed_effect_col=fixed_effect_col,
            train_mask=train_mask,
        )

        y = df[y_col]
        if train_mask is not None:
            X_df = X_df[train_mask]
            y = y[train_mask]

        valid_mask = y.notna()
        X_train = X_df[valid_mask]
        y_train = y[valid_mask]
        if X_train.empty or y_train.empty:
            raise ValueError("No data available to fit the linear regression (check feature/target columns).")

        model = SklearnLinearRegression().fit(X_train, y_train)

        params = {
            "coef_": model.coef_.tolist(),
            "intercept_": float(model.intercept_),
            "X_cols": X_df.columns.tolist(),  # full feature order including dummies
            "base_features": X_cols,
            "fill_values": fill_values.to_dict(),
            "fixed_effect_columns": fixed_effect_columns,
            "fixed_effect_col": fixed_effect_col,
        }
        name = kwargs.get("name", "linreg_pred")
        empty_preds = pd.DataFrame(index=outputs.index, columns=[name])
        return name, params, empty_preds

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        params = kwargs.get("model_params") or LinearRegression.load(
            kwargs["tick"], kwargs.get("tmp_dir", "./tmp/json/")
        )
        fixed_effect_col = kwargs.get("diff_ticks") or params.get("fixed_effect_col")
        base_features = params.get("base_features", params.get("X_cols", []))

        df, X_df, _, _ = LinearRegression._build_feature_matrix(
            X,
            base_features,
            fixed_effect_col=fixed_effect_col,
            train_mask=None,
        )

        if fixed_effect_col and fixed_effect_col in df.columns:
            dummies = pd.get_dummies(df[fixed_effect_col], prefix=fixed_effect_col, drop_first=True)
            dummies = dummies.reindex(columns=params.get("fixed_effect_columns", []), fill_value=0)
            X_df = pd.concat([X_df.drop(columns=[fixed_effect_col], errors="ignore"), dummies], axis=1)

        X_df = X_df.loc[:, ~X_df.columns.duplicated()]
        fill_values = pd.Series(params.get("fill_values", {}))
        if not fill_values.empty:
            X_df = X_df.fillna(fill_values)
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_df = X_df.reindex(columns=params["X_cols"], fill_value=0)

        coef = np.array(params["coef_"])
        intercept = float(params["intercept_"])
        name = kwargs.get("name", "linreg_pred")
        series = pd.Series(X_df.to_numpy() @ coef + intercept, index=df.index, name=name)
        return name, params, series.to_frame(name)

    @staticmethod
    @all_stocks
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
