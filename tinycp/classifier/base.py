from venn_abers import VennAbers
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
import warnings
import numpy as np
import sklearn.metrics
from abc import ABC, abstractmethod

# Suprimir o aviso específico
warnings.filterwarnings("ignore", category=RuntimeWarning, module="venn_abers")


class BaseOOBConformalClassifier(ABC):
    """
    BaseConformalClassifier

    A base class for conformal prediction using a RandomForestClassifier as the learner
    and Venn-Abers calibration for confidence estimation.
    This approach provides valid predictions with a specified significance level (alpha).

    Conformal classifiers aim to quantify uncertainty in predictions.
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
        scoring_func: str, default="bm"
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

        self.learner = learner
        self.alpha = alpha
        self.calibration_layer = VennAbers()

        # Ensure the learner is fitted
        check_is_fitted(learner, attributes=["oob_decision_function_"])

        if learner.n_classes_ > 2:
            raise ValueError("This classifier supports only binary classification.")

        self.feature_importances_ = learner.feature_importances_

        self.hinge = None
        self.n = None
        self.y = None

    @abstractmethod
    def fit(self, y):
        """
        Fits the classifier to the training data.
        """
        pass

    @abstractmethod
    def predict_set(self, X, alpha=None):
        """
        Generate a prediction set for the given input.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_qhat(self, ncscore, q_level):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
        """
        pass

    @abstractmethod
    def _compute_set(self, ncscore, qhat):
        """
        Compute a set based on the given ncscore and qhat.
        """
        pass

    def _compute_prediction(self, prediction_set):
        """
        Compute the prediction based on the given prediction set.

        This method evaluates each row in the prediction set and returns 1 if all elements
        in the row match the pattern [0, 1], otherwise returns 0.
        """
        return np.where(np.all(prediction_set == [0, 1], axis=1), 1, 0)

    def _bookmaker_informedness(self, y, y_pred):
        """
        Calculate the bookmaker informedness score for the given true and predicted labels.
        """
        return sklearn.metrics.balanced_accuracy_score(y, y_pred, adjusted=True)

    def _select_scoring_function(self, scoring_func):
        """
        Select the scoring function based on the provided string.
        """

        if scoring_func == "bm":
            func = self._bookmaker_informedness
        elif scoring_func == "mcc":
            func = sklearn.metrics.matthews_corrcoef
        else:
            raise ValueError("Invalid metric function. Please use 'bm' or 'mcc'.")
        return func

    def _get_alpha(self, alpha):
        """Helper to retrieve the alpha value."""
        return alpha or self.alpha

    def generate_non_conformity_score(self, y_prob):
        """
        Generates the non-conformity score based on the hinge loss.

        This function calculates the non-conformity score for conformal prediction
        using the hinge loss approach.
        """
        return 1 - y_prob

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

        alpha = self._get_alpha(alpha)

        q_level = np.ceil((self.n + 1) * (1 - alpha)) / self.n

        return self._compute_qhat(self.hinge, q_level)

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

    def calibrate(self, X, y, max_alpha=0.2, func="mcc"):
        """
        Calibrates the alpha value to minimize error rates.

        The method iterates over a range of alpha values (0.01 to `max_alpha`) to find the
        optimal significance level based on the specified metric function.

        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input samples for calibration.
        y: array-like of shape (n_samples,)
            True labels.
        max_alpha: float, default=0.2
            Maximum alpha value to consider during calibration.

        Raises:
        -------
        ValueError
            If an invalid metric function is provided.

        Returns:
        --------
        float
            The optimal alpha value.
        """

        scoring_func = self._select_scoring_function(func)

        alphas = {k: None for k in np.round(np.arange(0.01, max_alpha + 0.01, 0.01), 2)}

        for alpha in alphas:
            y_pred = self.predict(X, alpha)
            alphas[alpha] = scoring_func(y, y_pred)

        self.alpha = max(alphas, key=alphas.get)

        return self.alpha

    def predict(self, X, alpha=None):
        """
        Predicts the classes for the input samples.

        Parameters:
        -----------
        X: np.ndarray of shape (n_samples, n_features)
            Input samples.
        alpha: float, optional
            Significance level. If None, defaults to the classifier's alpha value.

        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Predicted class labels, where 1 indicates the model's certainty.
        """

        alpha = self._get_alpha(alpha)

        prediction_set = self.predict_set(X, alpha)

        return self._compute_prediction(prediction_set)

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

    def _evaluate_generalization(self, X, y, alpha=None):
        """
        Evaluate the generalization gap of the conformal classifier.

        The generalization gap quantifies the difference in performance
        between the training and test datasets. A smaller gap indicates
        better generalization to unseen data, while a larger gap suggests
        potential overfitting. It is calculated as the difference between
        the error rate on the training set and the error rate on the test set.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features of the test set.
        y : array-like of shape (n_samples,)
            True labels corresponding to the test set.
        alpha : float, optional
            Significance level for conformal prediction. If None, the value
            of `self.alpha` is used.

        Returns:
        -------
        float
            The generalization gap, calculated as:
            (Training Error - Test Error).
        """

        alpha = self._get_alpha(alpha)

        ncscore = self.generate_non_conformity_score(
            self.learner.oob_decision_function_
        )

        qhat = self.generate_conformal_quantile(alpha)

        prediction_set = self._compute_set(ncscore, qhat)
        y_pred = self._compute_prediction(prediction_set)

        training_error = 1 - sklearn.metrics.balanced_accuracy_score(
            y_pred, self.y, adjusted=False
        )
        test_error = 1 - sklearn.metrics.balanced_accuracy_score(
            self.predict(X, alpha), y, adjusted=False
        )
        return training_error - test_error

    def _shuffle(self, scores, n):
        """
        Shuffle the given scores and split them into calibration and validation sets.
        """
        np.random.shuffle(scores)  # shuffle
        calib_scores, val_scores = (scores[:n], scores[n:])  # split
        return calib_scores, val_scores

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

        alpha = self._get_alpha(alpha)

        coverages = np.zeros((iterations,))
        y_prob = self.predict_proba(X)
        scores = 1 - y_prob
        n = int(len(scores) * 0.20)

        for i in range(iterations):
            calib_scores, val_scores = self._shuffle(scores, n)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = self._compute_qhat(calib_scores, q_level)
            coverages[i] = self._compute_set(val_scores, qhat).mean()  # see caption
            average_coverage = coverages.mean()  # should be close to 1-alpha

        return average_coverage

    def evaluate(self, X, y, alpha=None):
        """
        Evaluate the classifier on the given dataset.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The true labels for X.
        alpha : float, optional
            The significance level for prediction sets. If None, a default value is used.

        Returns:
        --------
        results : dict
            A dictionary containing the evaluation metrics:
            - "alpha": The significance level used.
            - "empirical_coverage": The empirical coverage of the prediction sets.
            - "one_c": The proportion of prediction sets containing exactly one element.
            - "avg_c": The average size of the prediction sets.
            - "empty": The proportion of empty prediction sets.
            - "error": The classification error rate.
            - "log_loss": The log loss of the predictions.
            - "ece": The expected calibration error.
            - "bm": The bookmaker informedness.
            - "mcc": The Matthews correlation coefficient.
            - "f1": The F1 score.
            - "generalization": The generalization error.
        """

        alpha = self._get_alpha(alpha)

        # Helper function for rounding
        def rounded(value):
            return np.round(value, 3)

        # Predictions and probabilities
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)
        predict_set = self.predict_set(X, alpha)

        # Metrics calculation
        empirical_coverage = rounded(self._empirical_coverage(X, alpha))
        one_c = rounded(np.mean([np.sum(p) == 1 for p in predict_set]))
        avg_c = rounded(np.mean([np.sum(p) for p in predict_set]))
        empty = rounded(np.mean([np.sum(p) == 0 for p in predict_set]))
        error = rounded(1 - np.mean(predict_set[np.arange(len(y)), y]))
        log_loss = rounded(sklearn.metrics.log_loss(y, y_prob[:, 1]))
        ece = rounded(self._expected_calibration_error(y, y_prob))
        generalization = rounded(self._evaluate_generalization(X, y, alpha))
        bookmaker_informedness = rounded(self._bookmaker_informedness(y, y_pred))
        matthews_corr = rounded(sklearn.metrics.matthews_corrcoef(y, y_pred))
        f1 = rounded(sklearn.metrics.f1_score(y, self.predict(X, alpha)))

        # Results aggregation
        results = {
            "alpha": alpha,
            "empirical_coverage": empirical_coverage,
            "one_c": one_c,
            "avg_c": avg_c,
            "empty": empty,
            "error": error,
            "log_loss": log_loss,
            "ece": ece,
            "bm": bookmaker_informedness,
            "mcc": matthews_corr,
            "f1": f1,
            "generalization": generalization,
        }

        return results
