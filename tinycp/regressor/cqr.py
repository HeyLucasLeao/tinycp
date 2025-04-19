from sklearn.base import BaseEstimator
import numpy as np
from typing import List
from .base import BaseRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ConformalizedQuantileRegressor(BaseRegressor):
    """
    A conformalized quantile regressor that provides valid prediction intervals
    using a specified quantile regression model as the learner. It ensures statistical validity
    under the assumption of exchangeability, based on the conformalized quantile regression (CQR) method.
    Note:
    -----
    This class is designed to work with learner models such as those provided by
    the Quantile Forest library: https://github.com/zillow/quantile-forest
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
        quantiles: List[float] = [0.05, 0.95],
    ):
        """
        Initializes the conformalized regressor with a specified learner and significance level.
        Parameters:
        ----------
        learner : BaseEstimator
            The base learner to be used in the regressor.
        alpha : float, default=0.05
            The significance level applied in the regressor.
        """
        self.quantiles = quantiles
        super().__init__(learner, alpha)

    def fit(self, X, y, oob=False):
        """
        Fit the conformalized regressor by calculating nonconformity scores.

        Parameters:
        - X: Training feature matrix
        - y: Training target vector
        - quantiles: List of quantiles for prediction intervals
        - oob: Whether to use out-of-bag predictions (if supported by the learner)

        Returns:
        - self: The fitted conformalized regressor
        """
        if X is None or y is None:
            raise ValueError(
                "Both training data (X) and true labels (y) must be provided."
            )

        if oob:
            if not hasattr(self.learner, "oob_prediction_"):
                raise ValueError(
                    "OOB predictions are not available for the provided learner."
                )
            # Use out-of-bag predictions if available
            self.decision_function_ = self.learner.predict(
                X, quantiles=self.quantiles, oob_score=True
            )
        else:
            self.decision_function_ = self.learner.predict(X, quantiles=self.quantiles)

        self.n = len(self.decision_function_)
        self.ncscore = np.maximum(
            self.decision_function_[:, 0] - y, y - self.decision_function_[:, 1]
        )

        return self

    def predict_interval(self, X_test, alpha=None):
        """
        Generate prediction intervals for the given model and calibration data.
        """

        alpha = self._get_alpha(alpha)
        qhat = self.generate_conformal_quantile(alpha)
        y_pred = self.learner.predict(X_test)

        lower_bound = y_pred[:, 0] - qhat
        upper_bound = y_pred[:, 1] + qhat

        return np.array([lower_bound, upper_bound]).T
