from .linreg import LinearRegression as LinReg
from .logreg import LogisticRegression as LogReg
from .mlp_reg import MultiLayerPerceptron as MLP
from .gboost import GradientBoostingRegressor as GBReg

__all__ = [
    "LinReg",
    "LogReg",
    "MLP",
    "GBReg"
]