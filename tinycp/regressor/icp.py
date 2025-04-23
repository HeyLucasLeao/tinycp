from sklearn.base import BaseEstimator
import numpy as np
from .base import BaseConformalRegressor


class ConformalizedRegressor(BaseConformalRegressor):
    """
    ConformalizedRegressor
    This class implements a conformalized regressor that provides valid prediction intervals
    using a specified regression model as the learner. It is based on the split inductive conformal prediction (ICP) method,
    ensuring statistical validity under the assumption of exchangeability.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
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
        super().__init__(learner, alpha)

    def fit(self, X=None, y=None, oob=False):

        if y is None:
            raise ValueError("The true labels (y) must be provided.")
        if oob:
            if not hasattr(self.learner, "oob_prediction_"):
                raise ValueError(
                    "OOB predictions are not available for the provided learner."
                )
            self.decision_function_ = self.learner.oob_prediction_
        else:
            if X is None:
                raise ValueError(
                    "Training data (X) must be provided if OOB is not used."
                )

            self.decision_function_ = self.learner.predict(X)

        self.n = len(self.decision_function_)

        self.ncscore = np.abs(y - self.decision_function_)

        return self

    def predict_interval(self, X_test, alpha=None):
        """
        Generate prediction intervals for the given model and calibration data.
        """

        alpha = self._get_alpha(alpha)
        qhat = self.generate_conformal_quantile(alpha)
        y_pred = self.learner.predict(X_test)

        # Calculate the lower and upper bounds of the prediction intervals
        lower_bound = y_pred - qhat
        upper_bound = y_pred + qhat

        return np.array([lower_bound, upper_bound]).T
