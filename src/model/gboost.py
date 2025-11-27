"""
GradientBoosting (with target-encoded fixed effects) model implementation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class GradientBoostingRegressor(BaseModel):
    """
    Drop-in replacement for the old MLPRegressor wrapper.  We encode the
    fixed-effect column (diff_ticks) via TargetEncoder, scale the numeric
    features, and train a GradientBoostingRegressor on top.
    """

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model: Pipeline | None = None

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_pipeline(**kwargs) -> Pipeline:
        encoder_kwargs = kwargs.get("encoder_kwargs", {})
        scaler_kwargs = kwargs.get("scaler_kwargs", {})
        gbr_kwargs = kwargs.get("gbr_kwargs", {})

        return Pipeline([
            ("encoder", TargetEncoder(**encoder_kwargs)),
            ("scaler", StandardScaler(**scaler_kwargs)),
            ("model", GradientBoostingRegressor(**gbr_kwargs)),
        ])

    @staticmethod
    def serialise(**kwargs) -> None:
        tick = kwargs["tick"]
        tmp_dir = Path(kwargs.get("tmp_dir", "./tmp/json/"))
        params = kwargs["model_params"]

        payload = {
            "encoder": params["encoder"],
            "scaler_mean_": params["scaler_mean_"],
            "scaler_scale_": params["scaler_scale_"],
            "gbr_params": params["gbr_params"],
            "X_cols": params["X_cols"],
            "fixed_effect_col": params.get("fixed_effect_col"),
        }
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / f"{tick}_mlpreg.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load(tick: str, tmp_dir: str = "./tmp/json/") -> dict:
        return json.loads(Path(tmp_dir, f"{tick}_mlpreg.json").read_text())

    # --------------------------------------------------------------------- #
    # Fit / Predict / Fit-Predict
    # --------------------------------------------------------------------- #
    @staticmethod
    def fit(outputs, X_cols: list[str], y_col: str, **kwargs):
        name = kwargs.get("name", "gbm_pred")
        fixed_effect_col = kwargs.get("diff_ticks")
        X = outputs[X_cols].copy()
        y = outputs[y_col].values
        X = X.loc[:, ~X.columns.duplicated()]

        encoder_cols = []
        if fixed_effect_col and fixed_effect_col in outputs.columns:
            X = pd.concat([X, outputs[[fixed_effect_col]]], axis=1)
            encoder_cols = [fixed_effect_col]

        pipeline = GradientBoostingMLP._build_pipeline(**kwargs)
        pipeline.set_params(encoder__cols=encoder_cols)
        pipeline.fit(X, y)

        # stash components we need for standalone prediction
        encoder: TargetEncoder = pipeline.named_steps["encoder"]
        scaler: StandardScaler = pipeline.named_steps["scaler"]
        gbr: GradientBoostingRegressor = pipeline.named_steps["model"]

        params = {
            "encoder": {
                "cols": encoder.cols,
                "mapping": encoder.mapping_.to_dict(),
                "handle_missing": encoder.handle_missing,
                "handle_unknown": encoder.handle_unknown,
            },
            "scaler_mean_": scaler.mean_.tolist(),
            "scaler_scale_": scaler.scale_.tolist(),
            "gbr_params": {
                "init": gbr.init_.get_params() if gbr.init_ else None,
                "tree_params": gbr.get_params(),
                "estimators_": [tree.tree_.__getstate__() for tree in gbr.estimators_.ravel()],
            },
            "X_cols": X.columns.tolist(),
            "fixed_effect_col": fixed_effect_col,
        }

        empty_preds = pd.DataFrame(index=outputs.index, columns=[name])
        return name, params, empty_preds

    @staticmethod
    def _rebuild_encoder(encoder_payload: dict) -> TargetEncoder:
        encoder = TargetEncoder(cols=encoder_payload["cols"])
        encoder.mapping_ = {
            col: pd.Series(mapping) for col, mapping in encoder_payload["mapping"].items()
        }
        encoder.handle_missing = encoder_payload["handle_missing"]
        encoder.handle_unknown = encoder_payload["handle_unknown"]
        return encoder

    @staticmethod
    def _rebuild_model(gbr_payload: dict) -> GradientBoostingRegressor:
        gbr = GradientBoostingRegressor()
        gbr.set_params(**gbr_payload["tree_params"])
        if gbr_payload["init"]:
            gbr.init_.set_params(**gbr_payload["init"])
        # Reconstruct estimators_
        estimators = []
        for state in gbr_payload["estimators_"]:
            estimator = GradientBoostingRegressor().estimator_.__class__()
            estimator.tree_.__setstate__(state)
            estimators.append(estimator)
        array_est = np.array(estimators, dtype=object).reshape(-1, 1)
        gbr.estimators_ = array_est
        return gbr

    @staticmethod
    def predict(X: pd.DataFrame, **kwargs):
        params = kwargs.get("model_params") or GradientBoostingMLP.load(
            kwargs["tick"], kwargs.get("tmp_dir", "./tmp/json/")
        )
        name = kwargs.get("name", "gbm_pred")

        fixed_effect_col = kwargs.get("diff_ticks") or params.get("fixed_effect_col")
        X_pred = X.copy()
        X_pred = X_pred.loc[:, ~X_pred.columns.duplicated()]
        if fixed_effect_col and fixed_effect_col in X_pred.columns:
            # keep the raw categorical column for the encoder
            pass

        X_pred = X_pred.reindex(columns=params["X_cols"], fill_value=0)

        # rebuild encoder + scaler + model
        encoder = GradientBoostingMLP._rebuild_encoder(params["encoder"])
        scaler = StandardScaler()
        scaler.mean_ = np.array(params["scaler_mean_"])
        scaler.scale_ = np.array(params["scaler_scale_"])
        gbr = GradientBoostingRegressor(**params["gbr_params"]["tree_params"])

        encoded = encoder.transform(X_pred.copy())
        scaled = (encoded - scaler.mean_) / scaler.scale_
        preds = pd.Series(gbr.predict(scaled), index=X.index, name=name)
        return name, params, preds.to_frame(name)

    @staticmethod
    def fit_predict(outputs, **kwargs):
        name, params, _ = GradientBoostingMLP.fit(outputs, **kwargs)
        _, _, preds = GradientBoostingMLP.predict(outputs, model_params=params, **kwargs)

        if kwargs.get("save_params"):
            GradientBoostingMLP.serialise(
                tick=kwargs.get("tick", ""),
                tmp_dir=kwargs.get("tmp_dir", "./tmp/json/"),
                model_params=params,
            )

        return name, params, preds
