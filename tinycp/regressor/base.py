from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error


class BaseRegressor(ABC):
    """
    BaseRegressor

    A base class for conformal regression using a model as the learner
    to provide valid prediction intervals with a specified significance level (alpha).

    Conformal regressors aim to quantify uncertainty in predictions by generating
    prediction intervals that adapt to the data and model.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
    ):
        """
        Initializes the regressor with a specified learner and significance level.

        Parameters:
        ----------
        learner : BaseEstimator
            The base learner to be used in the regressor.
        alpha : float, default=0.05
            The significance level applied in the regressor.

        Attributes:
        ----------
        learner : BaseEstimator
            The base learner employed in the regressor.
        alpha : float
            The significance level applied in the regressor.
        decision_function_ : array-like, default=None
            The decision function values after fitting the model.
        ncscore : array-like, default=None
            Nonconformity scores used for conformal prediction.
        n : int, default=None
            Number of calibration samples.
        """

        self.learner = learner
        self.alpha = alpha
        self.decision_function_ = None
        self.ncscore = None
        self.n = None

        # Ensure the learner is fitted
        check_is_fitted(learner)

    @abstractmethod
    def fit(self, y):
        """
        Fits the classifier to the training data.
        """
        pass

    @abstractmethod
    def predict_interval(self, X, alpha=None):
        """
        Generate prediction intervals for the input data.
        To be implemented by subclasses.
        """
        pass

    def _compute_qhat(self, ncscore, q_level):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
        """

        return np.quantile(ncscore, q_level, method="higher")

    def _get_alpha(self, alpha):
        """Helper to retrieve the alpha value."""
        return alpha or self.alpha

    def generate_conformal_quantile(self, alpha=None):
        """
        Generate the conformal quantile for conformal prediction.

        This method calculates the conformal quantile based on the nonconformity scores
        of the calibration samples. The quantile serves as a threshold to determine
        the prediction intervals in conformal prediction.

        Parameters:
        -----------
        alpha : float, optional
            The significance level for conformal prediction. If None, the default
            value of self.alpha is used.

        Returns:
        --------
        float
            The computed conformal quantile.

        Notes:
        ------
        - The quantile is computed as ceil((n + 1) * (1 - alpha)) / n, where n is the
          number of calibration samples.
        - This method relies on the self.ncscore attribute, which should contain the
          nonconformity scores of the calibration samples.
        """

        alpha = self._get_alpha(alpha)

        q_level = np.ceil((self.n + 1) * (1 - alpha)) / self.n

        return self._compute_qhat(self.ncscore, q_level)

    def _coverage_rate(self, X, y, alpha=None):
        """
        Evaluate coverage of prediction intervals on test data.

        Parameters:
        - X_test: test feature matrix
        - y_test: test target vector

        Returns:
        - coverage: proportion of test points within prediction intervals
        """
        alpha = self._get_alpha(alpha)
        y_pred = self.predict_interval(X, alpha)
        coverages = (y >= y_pred[:, 0]) & (y <= y_pred[:, 1])

        return np.mean(coverages)

    def predict(self, X_test, alpha=None):
        """
        Generate prediction intervals for the given model and calibration data.
        """

        alpha = self._get_alpha(alpha)
        y_pred = self.predict_interval(X_test, alpha)

        return np.sum(y_pred, axis=1) / 2

    def interval_width_mean(self, X):
        """
        Calcula a amplitude média dos intervalos de predição.

        Parâmetros:
        intervals (numpy.ndarray): Um array contendo os intervalos de predição,
                                    onde cada linha representa [limite_inferior, limite_superior].

        Retorna:
        float: A amplitude média dos intervalos.
        """
        y_pred = self.predict_interval(X)
        widths = y_pred[:, 1] - y_pred[:, 0]
        return np.mean(widths)

    def evaluate(self, X, y, alpha=None):

        alpha = self._get_alpha(alpha)

        # Helper function for rounding
        def rounded(value):
            return np.round(value, 3)

        # Metrics calculation
        total = len(X)
        coverage_rate = rounded(self._coverage_rate(X, y, alpha))
        interval_width_mean = rounded(self.interval_width_mean(X))
        y_pred = self.predict(X, alpha)
        mae = rounded(mean_absolute_error(y, y_pred))
        mbe = rounded(np.mean(y_pred - y))

        results = {
            "total": total,
            "alpha": alpha,
            "coverage_rate": coverage_rate,
            "interval_width_mean": interval_width_mean,
            "mae": mae,
            "mbe": mbe,
        }

        return results
