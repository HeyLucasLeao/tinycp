from venn_abers import VennAbers
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
import warnings
import numpy as np
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    f1_score,
    balanced_accuracy_score,
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

    def calibrate(self, X, y, minimal=0.2, func=balanced_accuracy_score):
        """
        Calibrates the alpha value to minimize the error rate
        using Cost Sensitive Learning methodology, using balanced_accuracy_score.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The test input samples.
        y: array-like of shape (n_samples,)
            The true labels for X.
        minimal: float
            The minimal value of alpha to consider.

        For each alpha value (0.01, 0.02, ..., minimal), we do the following:
        - Calculate predictions y_pred using the self.predict(X, alpha) function.

        Returns:
            The updated instance (self.alpha) with the calibrated alpha value.
        """

        alphas = {k: None for k in np.round(np.arange(0.01, minimal + 0.01, 0.01), 2)}

        for alpha in alphas:
            y_pred = self.predict(X, alpha)
            alphas[alpha] = func(y, y_pred)

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

    def _expected_calibration_error(self, y, y_prob, M=5):
        """
        Generate the expected calibration error (ECE) of the classifier.

        Parameters:
        y: array-like of shape (n_samples,)
            The true labels.
        y_prob: array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        M: int, default=5
            The number of bins for the uniform binning approach.

        Returns:
        ece: float
            The expected calibration error.

        The function works as follows:
        - It first creates M bins with uniform width over the interval [0, 1].
        - For each sample, it computes the maximum predicted probability and makes a prediction.
        - It then checks whether each prediction is correct or not.
        - For each bin, it calculates the empirical probability of a sample falling into the bin.
        - If the empirical probability is greater than 0, it computes the accuracy and average confidence of the bin.
        - It then calculates the absolute difference between the accuracy and the average confidence, multiplies it by the empirical probability, and adds it to the total ECE.
        """

        # uniform binning approach with M number of bins
        bin_boundaries = np.linspace(0, 1, M + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # get max probability per sample i
        confidences = np.max(y_prob, axis=1)
        # get predictions from confidences (positional in this case)
        predicted_label = np.argmax(y_prob, axis=1)

        # get a boolean list of correct/false predictions
        predictions = predicted_label == y

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            in_bin = np.logical_and(
                confidences > bin_lower.item(), confidences <= bin_upper.item()
            )
            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prob_in_bin = np.mean(in_bin)

            if prob_in_bin > 0:
                # get the accuracy of bin m: acc(Bm)
                avg_pred = np.mean(predictions[in_bin])
                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                ece += np.abs(avg_pred - avg_confidence_in_bin) * prob_in_bin
        return ece

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
        matthews_corr = rounded(balanced_accuracy_score(y, y_pred))
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
            "balanced_accuracy_score": matthews_corr,
            "f1_score": f1,
            "alpha": alpha,
        }

        return pd.DataFrame([results])
