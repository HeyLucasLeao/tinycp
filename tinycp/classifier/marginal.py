from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
from .base import BaseOOBConformalClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning, module="venn_abers")


class OOBBinaryMarginalConformalClassifier(BaseOOBConformalClassifier):
    """
    Conformal classifier based on Out-of-Bag (OOB) predictions.
    Uses RandomForestClassifier and Venn-Abers calibration.
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
        alpha: float, default=0.05
            The significance level applied in the classifier.

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

        super().__init__(learner, alpha)

    def fit(self, y):
        """
        Fits the classifier to the training data. Calculates the conformity score for each training instance.

        Parameters:
        y: array-like of shape (n_samples,)
            The true labels.

        Returns:
        self: object
            Returns self.

        The function works as follows:
        - It first gets the out-of-bag probability predictions from the learner.
        - It then fits the calibration layer to these predictions and the true labels.
        - It computes the probability for each instance.
        - It finally turns these probabilities into non-conformity measure.
        """

        # Get the probability predictions
        self.n = len(self.learner.oob_decision_function_)
        y_prob = self.learner.oob_decision_function_

        self.calibration_layer.fit(y_prob, y)
        y_prob, _ = self.calibration_layer.predict_proba(y_prob)

        # We only need the probability for the true class
        y_prob = y_prob[np.arange(self.n), y]

        self.hinge = self.generate_non_conformity_score(y_prob)
        self.y = y

        return self

    def _compute_qhat(self, ncscore, q_level):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
        """
        return np.quantile(ncscore, q_level, method="higher")

    def _compute_set(self, ncscore, qhat):
        """
        Compute a predict set based on the given ncscore and qhat.
        """
        return (ncscore <= qhat).astype(int)

    def predict_set(self, X, alpha=None):
        """
        Predicts the possible set of classes for the instances in X based on the predefined significance level.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.
        alpha: float, default=None
            The significance level. If None, the value of self.alpha is used.

        Returns:
        prediction_set: array-like of shape (n_samples, n_classes)
            The predicted set of classes. A class is included in the set if its non-conformity score is less
            than or equal to the quantile of the hinge loss distribution at the (n+1)*(1-alpha)/n level.
        """

        alpha = self._get_alpha(alpha)

        y_prob = self.predict_proba(X)
        ncscore = self.generate_non_conformity_score(y_prob)
        qhat = self.generate_conformal_quantile(alpha)

        return self._compute_set(ncscore, qhat)

    def predict_p(self, X):
        """
        Calculate the p-values for each instance in the input data X using a non-conformity score.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data for which the p-values need to be predicted.

        Returns:
        --------
        p_values : array-like of shape (n_samples, n_classes)
            The p-values for each instance in X for each class.

        """
        y_prob = self.predict_proba(X)
        ncscore = self.generate_non_conformity_score(y_prob)
        p_values = np.zeros_like(ncscore)

        for i in range(ncscore.shape[0]):
            for j in range(ncscore.shape[1]):
                numerator = np.sum(self.hinge >= ncscore[i][j]) + 1
                denumerator = self.n + 1
                p_values[i, j] = numerator / denumerator

        return p_values
