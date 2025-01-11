# TinyCP
TinyCP is an experimental Python library for conformal predictions, providing tools to generate valid prediction sets with a specified significance level (alpha). This project aims to facilitate the implementation of personal and future projects on the topic.

For more information on a previous project related to Out-of-Bag (OOB) solutions, visit [this link](https://github.com/HeyLucasLeao/cp-study).

## Changes about previous work
- `calibrate`: instead of `Balanced Accuracy Score`, it can be calibrated either `Matthews Correlation Coefficient` or `Bookmaker Informedness Score`, for better reliability
- `evaluate`: scores `bm` and `mcc` for more reliability

Currently, TinyCP supports Out-of-Bag (OOB) solutions for `RandomForestClassifier` in binary classification problems. For more options and advanced features, consider exploring [Crepes](https://github.com/henrikbostrom/crepes).

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

### Importing Classes

Import the conformal classifiers from the `tinycp.classifier` module:

```python
from tinycp.classifier.class_conditional import OOBBinaryClassConditionalConformalClassifier
from tinycp.classifier.marginal import OOBBinaryMarginalConformalClassifier
```

### Example

Example usage of `OOBBinaryClassConditionalConformalClassifier`:

```python
from sklearn.ensemble import RandomForestClassifier
from tinycp.classifier.class_conditional import OOBBinaryClassConditionalConformalClassifier

# Create and fit a RandomForestClassifier
learner = RandomForestClassifier(n_estimators=100, oob_score=True)
X_train, y_train = ...  # your training data
learner.fit(X_train, y_train)

# Create and fit the conformal classifier
conformal_classifier = OOBBinaryClassConditionalConformalClassifier(learner)
conformal_classifier.fit(y_train)

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

### BaseConformalClassifier

`BaseConformalClassifier` is a base class for conformal prediction using a RandomForestClassifier and Venn-Abers calibration for confidence estimation.

### OOBBinaryClassConditionalConformalClassifier

`OOBBinaryClassConditionalConformalClassifier` is a class conditional conformal classifier based on OOB methodology, using a random forest classifier as the learner.

### OOBBinaryMarginalConformalClassifier

`OOBBinaryMarginalConformalClassifier` is a conformal classifier based on OOB predictions, using RandomForestClassifier and Venn-Abers calibration.

## License

This project is licensed under the MIT License.
