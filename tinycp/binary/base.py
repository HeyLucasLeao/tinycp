from venn_abers import VennAbers
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
import warnings
import numpy as np
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
)
import pandas as pd

# Suprimir o aviso específico
warnings.filterwarnings("ignore", category=RuntimeWarning, module="venn_abers")


class BaseConformalClassifier:
    """
    A modrian class conditional conformal classifier based on Out-of-Bag (OOB) methodology, utilizing a random forest classifier as the underlying learner.
    This class is inspired by the WrapperClassifier classes from the Crepes library.
    """

    def __init__(
        self,
        learner: RandomForestClassifier,
        alpha: float = 0.05,
    ):
        """
        Constructs the classifier with a specified learner and a Venn-Abers calibration layer.

        Parameters:
        learner: RandomForestClassifier
            The base learner to be used in the classifier.

        Attributes:
        learner: RandomForestClassifier
            The base learner employed in the classifier.
        calibration_layer: VennAbers
            The calibration layer utilized in the classifier.
        feature_importances_: array-like of shape (n_features,)
            The feature importances derived from the learner.
        hinge : array-like of shape (n_samples,), default=None
            Nonconformity scores based on the predicted probabilities. Measures the confidence margin
            between the predicted probability of the true class and the most likely incorrect class.
        alpha: float, default=0.05
            The significance level applied in the classifier.
        """

        self.learner = learner
        self.alpha = alpha
        self.calibration_layer = VennAbers()

        # Ensure the learner is fitted
        check_is_fitted(learner, attributes=["oob_decision_function_"])

        if learner.n_classes_ > 2:
            raise ("Learner has more than 2 labels.")

        self.feature_importances_ = learner.feature_importances_

        self.hinge = None
        self.n = None
        self.y = None

    def generate_non_conformity_score(self, y_prob):
        """
        Generates the non-conformity score based on the hinge loss.

        This function calculates the non-conformity score for conformal prediction
        using the hinge loss approach.

        Parameters:
        -----------
        y_prob : array-like of shape (n_samples,) or (n_samples, n_classes)
            The predicted probabilities for each class.

        Returns:
        --------
        array-like
            The non-conformity scores, where higher values indicate greater
            non-conformity.

        Notes:
        ------
        - This implementation assumes that y_prob contains probabilities and
          not raw model outputs.

        """
        return 1 - y_prob

    def predict_proba(self, X):
        """
        Predicts the class probabilities for the instances in X.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        p_prime: array-like of shape (n_samples, n_classes)
            The calibrated class probabilities.
        """

        y_score = self.learner.predict_proba(X)
        p_prime, _ = self.calibration_layer.predict_proba(y_score)
        return p_prime

    def calibrate(self, X, y):
        """
        Calibrates the alpha value to minimize the error rate
        using Cost Sensitive Learning methodology, using balanced_accuracy_score.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The test input samples.
        y: array-like of shape (n_samples,)
            The true labels for X.

        For each alpha value (0.10, 0.09, …, 0.01), we do the following:
        - Calculate predictions y_pred using the self.predict(X, alpha) function.

        Returns:
            The updated instance (self.alpha) with the calibrated alpha value.
        """

        alphas = {
            round(
                k,
                2,
            ): None
            for k in np.linspace(0.01, 0.10, 10)
        }

        for alpha in alphas:
            y_pred = self.predict(X, alpha)
            alphas[alpha] = matthews_corrcoef(y, y_pred)

        self.alpha = max(alphas, key=alphas.get)

        return self.alpha

    def predict(self, X, alpha=None):
        """
        Predicts the classes for the instances in X.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        predictions: array-like of shape (n_samples,)
            A predicted true class if the model has certainty based on the predefined significance level.
        """

        alpha = self.alpha if alpha is None else alpha

        y_pred = self.predict_set(X, alpha)

        return np.where(np.all(y_pred == [0, 1], axis=1), 1, 0)

    def evaluate(self, X, y, alpha=None):
        """
        Evaluates the performance of the conformal classifier on the given test data and labels.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The test input samples.
        y: array-like of shape (n_samples,)
            The true labels for X.
        alpha: float, default=None
            The significance level. If None, the value of self.alpha is used.

        Returns:
        pd.DataFrame
            A DataFrame containing the evaluation metrics.
        """
        alpha = alpha if alpha is not None else self.alpha

        # Helper function for rounding
        def rounded(value):
            return round(value, 3)

        # Predictions and probabilities
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)
        predict_set = self.predict_set(X, alpha)

        # Metrics calculation
        one_c = rounded(np.mean([np.sum(p) == 1 for p in predict_set]))
        avg_c = rounded(np.mean([np.sum(p) for p in predict_set]))
        empty = rounded(np.mean([np.sum(p) == 0 for p in predict_set]))
        error = rounded(1 - np.mean(predict_set[np.arange(len(y)), y]))
        log_loss_value = rounded(log_loss(y, y_prob[:, 1]))
        brier_loss_value = rounded(brier_score_loss(y, y_prob[:, 1]))
        ece = rounded(self._expected_calibration_error(y, y_prob))
        empirical_coverage = rounded(self._empirical_coverage(X, alpha))
        generalization = rounded(self._evaluate_generalization(X, y, alpha))
        matthews_corr = rounded(matthews_corrcoef(y, y_pred))
        f1 = rounded(f1_score(y, self.predict(X, alpha)))

        # Results aggregation
        results = {
            "one_c": one_c,
            "avg_c": avg_c,
            "empty": empty,
            "error": error,
            "log_loss": log_loss_value,
            "brier_loss": brier_loss_value,
            "ece": ece,
            "empirical_coverage": empirical_coverage,
            "generalization": generalization,
            "matthews_corrcoef": matthews_corr,
            "f1_score": f1,
            "alpha": alpha,
        }

        return pd.DataFrame([results])
