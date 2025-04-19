# TinyCP
TinyCP is an experimental Python library for conformal predictions, providing tools to generate valid prediction sets with a specified significance level (alpha). This project aims to facilitate the implementation of personal and future projects on the topic.

For more information on a previous project related to Out-of-Bag (OOB) solutions, visit [this link](https://github.com/HeyLucasLeao/cp-study).

## Changes about previous work
- `calibrate`: instead of `Balanced Accuracy Score`, it can be calibrated either `Matthews Correlation Coefficient` or `Bookmaker Informedness Score`, for better reliability.
- `evaluate`: scores `bm` and `mcc` for more reliability.

Currently, TinyCP supports Out-of-Bag (OOB) solutions for `RandomForestClassifier` in binary classification problems, as well as `RandomForestRegressor` and `RandomForestQuantileRegressor` for regression tasks. For additional options and advanced features, you may want to explore [Crepes](https://github.com/henrikbostrom/crepes).

## Installation

Install TinyCP using pip:

```bash
pip install tinycp
```

> **Note:** If you want to enable plotting capabilities, you need to install the extras using Poetry:

```bash
poetry install --E plot
```

## Usage

### Importing Classifiers

Import the conformal classifiers from the `tinycp.classifier` module:

```python
from tinycp.classifier import BinaryClassConditionalConformalClassifier
from tinycp.classifier import BinaryMarginalConformalClassifier
```
### Importing Regressors

Import the conformal regressors from the `tinycp.regressor` module:

```python
from tinycp.regressor import ConformalizedRegressor
from tinycp.regressor import ConformalizedQuantileRegressor
```
### Example

Example usage of `BinaryClassConditionalConformalClassifier`:

```python
from sklearn.ensemble import RandomForestClassifier
from tinycp.classifier import BinaryClassConditionalConformalClassifier

# Create and fit a RandomForestClassifier
learner = RandomForestClassifier(n_estimators=100, oob_score=True)
X_train, y_train = ...  # your training data
learner.fit(X_train, y_train)

# Create and fit the conformal classifier
conformal_classifier = BinaryClassConditionalConformalClassifier(learner)
conformal_classifier.fit(y=y_train, oob=True)

# Make predictions
X_test = ...  # your test data
predictions = conformal_classifier.predict(X_test)
```

### Evaluating the Classifier

Evaluate the performance of the conformal classifier using the `evaluate` method:

```python
results = conformal_classifier.evaluate(X_test, y_test)
print(results)
```

## Classes

### BinaryMarginalConformalClassifier

`BinaryMarginalConformalClassifier` A marginal coverage conformal classifier methodology utilizing a classifier as the underlying learner. This classifier supports the option to use Out-of-Bag (OOB) samples for calibration.


### BinaryClassConditionalConformalClassifier

`BinaryClassConditionalConformalClassifier` A class conditional conformal classifier methodology utilizing a classifier as the underlying learner. This classifier supports the option to use Out-of-Bag (OOB) samples for calibration.

### ConformalizedRegressor

`ConformalizedRegressor` A conformal regressor methodology utilizing a regressor as the underlying learner. This regressor supports the option to use Out-of-Bag (OOB) samples for calibration, providing valid prediction intervals for regression tasks.

### ConformalizedQuantileRegressor

`ConformalizedQuantileRegressor` A conformal quantile regressor methodology utilizing a quantile regressor as the underlying learner. This regressor supports the option to use Out-of-Bag (OOB) samples for calibration, offering more robust prediction intervals by leveraging quantile estimates.

## License

This project is licensed under the MIT License.
