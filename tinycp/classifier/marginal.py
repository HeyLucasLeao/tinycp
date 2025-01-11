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
        scoring_func: str = "mcc",
    ):
        """
        Constructs the classifier with a specified learner and a Venn-Abers calibration layer.

        Parameters:
        learner: RandomForestClassifier
            The base learner to be used in the classifier.
        alpha: float, default=0.05
            The significance level applied in the classifier.
        scoring_func: str, default="mcc"
            Scoring function to optimize. Acceptable values are:
            - "bm": Bookmaker Informedness
            - "mcc": Matthews Correlation Coefficient

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

        super().__init__(learner, alpha, scoring_func)

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

        alpha = alpha or self.alpha

        y_pred = self.predict_set(X, alpha)

        return np.where(np.all(y_pred == [0, 1], axis=1), 1, 0)

    def generate_conformal_quantile(self, alpha=None):
        """
        Generates the conformal quantile for conformal prediction.

        This function calculates the conformal quantile based on the non-conformity scores
        of the true label probabilities. The quantile is used as a threshold
        to determine the prediction set in conformal prediction.

        Parameters:
        -----------
        alpha : float, optional
            The significance level for conformal prediction. If None, uses the value
            of self.alpha.

        Returns:
        --------
        float
            The calculated conformal quantile.

        Notes:
        ------
        - The quantile is calculated as the (n+1)*(1-alpha)/n percentile of the non-conformity
          scores, where n is the number of calibration samples.
        - This method uses the self.hinge attribute, which should contain the non-conformity
          scores of the calibration samples.

        """

        alpha = alpha or self.alpha

        q_level = np.ceil((self.n + 1) * (1 - alpha)) / self.n
        qhat = np.quantile(self.hinge, q_level, method="higher")
        return qhat

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

        alpha = alpha or self.alpha

        y_prob = self.predict_proba(X)
        nc_score = self.generate_non_conformity_score(y_prob)
        qhat = self.generate_conformal_quantile(alpha)

        return (nc_score <= qhat).astype(int)

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
        nc_score = self.generate_non_conformity_score(y_prob)
        p_values = np.zeros_like(nc_score)

        for i in range(nc_score.shape[0]):
            for j in range(nc_score.shape[1]):
                numerator = np.sum(self.hinge >= nc_score[i][j]) + 1
                denumerator = self.n + 1
                p_values[i, j] = numerator / denumerator

        return p_values

    def _empirical_coverage(self, X, alpha=None, iterations=100):
        """
        Generate the empirical coverage of the classifier.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.
        alpha: float, default=None
            The significance level. If None, the value of self.alpha is used.
        iterations: int, default=100
            The number of iterations for the empirical coverage calculation.

        Returns:
        average_coverage: float
            The average coverage over the iterations. It should be close to 1-alpha.
        """

        alpha = alpha or self.alpha

        coverages = np.zeros((iterations,))
        y_prob = self.predict_proba(X)
        scores = 1 - y_prob
        n = int(len(scores) * 0.20)

        for i in range(iterations):
            np.random.shuffle(scores)  # shuffle
            calib_scores, val_scores = (scores[:n], scores[n:])  # split
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(calib_scores, q_level, method="higher")  # calibrate
            coverages[i] = (val_scores <= qhat).astype(float).mean()  # see caption
            average_coverage = coverages.mean()  # should be close to 1-alpha

        return average_coverage

    def _evaluate_generalization(self, X, y, alpha=None):
        """
        Measure the generalization gap of the model.

        The generalization gap indicates how well the model generalizes
        to unseen data. It is calculated as the difference between the
        error on the training set and the error on the test set.

        Parameters:
        X (array-like): Features of the test set
        y (array-like): Labels of the test set
        alpha (float, optional): Significance level for conformal prediction.
                                 If None, uses the default value.

        Returns:
        float: The generalization gap

        """

        alpha = alpha or self.alpha

        nc_score = self.generate_non_conformity_score(
            self.learner.oob_decision_function_
        )

        qhat = self.generate_conformal_quantile(alpha)

        y_pred = np.where(
            np.all((nc_score <= qhat).astype(int) == [0, 1], axis=1), 1, 0
        )

        training_error = 1 - self.scoring_func(y_pred, self.y)
        test_error = 1 - self.scoring_func(self.predict(X, alpha), y)
        return training_error - test_error
