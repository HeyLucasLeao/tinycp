import numpy as np
from typing import List
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from scipy.stats import beta


def efficiency_curve(clf, X: np.ndarray, fig_type=None, width=800, height=400):
    """
    Generates an efficiency and validity curve for a classifier.

    Args:
        clf (object): The classifier model.
        X (np.ndarray): Input data.
        fig_type (str, optional): Type of figure to display (e.g., 'png', 'svg'). Defaults to None.

    Returns:
        A efficiency curve plot.
    """

    def get_error_metrics(clf, X: np.ndarray) -> List:
        """
        Calculates error metrics for different error rates.

        Args:
            clf (object): The classifier model.
            X (np.ndarray): Input data.

        Returns:
            Tuple: Arrays for error_rate, efficiency_rate, and validity_rate.
        """

        error_rate = np.asarray([0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05])
        efficiency_rate = np.zeros(error_rate.shape)
        validity_rate = np.zeros(error_rate.shape)
        for i, error in enumerate(error_rate):
            predict_set = clf.predict_set(X, alpha=error)
            efficiency_rate[i] = np.sum([np.sum(p) == 1 for p in predict_set]) / len(
                predict_set
            )
            validity_rate[i] = np.sum(predict_set) / len(predict_set)
        return error_rate, efficiency_rate, validity_rate

    error_rate, efficiency_rate, validity_rate = get_error_metrics(clf, X)

    fig = go.Figure()

    # Create the bar chart
    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=efficiency_rate,
            mode="lines+markers",
            name="efficicency",
            line=dict(color="darkblue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=validity_rate,
            mode="lines+markers",
            name="validity",
            line=dict(color="orange"),
        )
    )

    fig.update_layout(
        title="Efficiency & Validity Curve",
        xaxis_title="Error Rate",
        yaxis_title="Score",
        legend=dict(title="Metric"),
        width=width,
        height=height,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    return fig.show(fig_type)


def reliability_curve(
    clf, X, y, n_bins=15, fig_type=None, model_name="RandomForest"
) -> go.Figure:
    """
    Generates a reliability curve for a classifier.

    Args:
        clf (object): The classifier model.
        X (np.ndarray): Input data.
        y (np.ndarray): True labels.
        n_bins (int, optional): Number of bins for the reliability curve. Defaults to 15.
        fig_type (str, optional): Type of figure to display (e.g., 'png', 'svg'). Defaults to None.

    Returns:
        go.Figure: Reliability curve plot.
    """

    y_prob = clf.predict_proba(X)[:, 1]

    v_prob_true, v_prob_pred = calibration_curve(
        y, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig = go.Figure()

    # Add traces for each model

    fig.add_trace(
        go.Scatter(x=v_prob_pred, y=v_prob_true, mode="lines+markers", name=model_name)
    )

    # Add a trace for the perfectly calibrated line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line=dict(dash="dash", color="grey"),
        )
    )

    fig.update_layout(
        title="Reliability Curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        legend_title="Model",
        autosize=False,
    )

    return fig.show(fig_type)


def histogram(clf, X, nbins=15, fig_type=None):
    """
    Generates a histogram of predicted scores for a classifier.

    Args:
        clf (object): The classifier model.
        X (np.ndarray): Input data.
        nbins (int, optional): Number of bins for the histogram. Defaults to 15.
        fig_type (str, optional): Type of figure to display (e.g., 'png', 'svg'). Defaults to None.

    Returns:
        A histogram plot.
    """
    y_prob = clf.predict_proba(X)[:, 1]
    fig = px.histogram(y_prob, nbins=nbins)
    fig.update_layout(
        title="Histogram of Predicted Scores",
        xaxis_title="Predicted Scores",
        yaxis_title="Count",
        legend_title="Modelos",
        autosize=False,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    fig.update_layout(showlegend=False)
    return fig.show(fig_type)


def confusion_matrix(clf, X, y, alpha=None, fig_type=None, percentage_by_class=True):
    """
    Generates an annotated heatmap of the confusion matrix for a classifier.

    Args:
        clf: Classifier object (e.g., sklearn classifier).
        X: Input features.
        y: True labels.
        alpha: Optional parameter for classifier prediction.
        fig_type: Optional figure type (e.g., 'png', 'svg').
        percentage_by_class: If True, displays percentages by class; otherwise, overall percentages.

    Returns:
        Annotated heatmap of the confusion matrix.
    """

    y_pred = clf.predict(X, alpha)
    tn, fp, fn, tp = sklearn_confusion_matrix(y, y_pred).ravel()
    labels = np.array([["FN", "TN"], ["TP", "FP"]])
    cm = np.array([[fn, tn], [tp, fp]])

    if percentage_by_class:
        total = cm.sum(axis=0)
        percentage = cm / total * 100
    else:
        percentage = cm / np.sum(cm) * 100

    annotation_text = np.empty_like(percentage, dtype="U10")

    for i in range(percentage.shape[0]):
        for j in range(percentage.shape[1]):
            annotation_text[i, j] = f"{labels[i, j]} {percentage[i, j]:.2f}"

    fig = ff.create_annotated_heatmap(
        cm,
        x=["Positive", "Negative"],
        y=["Negative", "Positive"],
        colorscale="Blues",
        hoverinfo="z",
        annotation_text=annotation_text,
    )

    fig.update_layout(width=400, height=400, title="Confusion Matrix")
    return fig.show(fig_type)


def beta_pdf_with_cdf_fill(alpha, beta_param, fig_type=None, start=0, end=1.0):
    """
    Plot the Beta Probability Density Function (PDF) with an optional fill between a specified interval,
    and display the cumulative density from the CDF as text.

    Parameters:
    alpha (int or float): The alpha (α) parameter of the Beta distribution.
    beta_param (int or float): The beta (β) parameter of the Beta distribution.
    fig_type: Optional figure type (e.g., 'png', 'svg').
    start (float): The starting value of the interval to fill. Default is 0.
    end (float): The ending value of the interval to fill. Default is 1.0.

    Returns:
    A Plotly figure displaying the Beta PDF with the specified filled interval and annotated cumulative density.
    """

    x = np.linspace(0, 1, 1000)
    y_pdf = beta.pdf(x, alpha, beta_param)

    fill_indices = (x >= start) & (x <= end)
    x_fill = x[fill_indices]
    y_pdf_fill = y_pdf[fill_indices]

    cumulative_density = beta.cdf(end, alpha, beta_param) - beta.cdf(
        start, alpha, beta_param
    )

    trace_pdf = go.Scatter(
        x=x, y=y_pdf, mode="lines", name=f"Beta PDF(α={alpha}, β={beta_param})"
    )
    trace_fill = go.Scatter(
        x=x_fill,
        y=y_pdf_fill,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.2)",
        name=f"Interval [{start}, {end}]",
    )

    layout = go.Layout(
        title="Beta PDF with CDF Fill",
        xaxis=dict(title="Success Rate", range=[min(x_fill) - 0.1, 1]),
        yaxis=dict(title="Density"),
        annotations=[
            dict(
                x=(start + end) / 2 if len(y_pdf_fill) > 0 else start * 1.1,
                y=max(y_pdf_fill) * 1.1,
                xref="x",
                yref="y",
                text=f"Cumulative Density: {cumulative_density:.2f}",
                showarrow=False,
                opacity=0.8,
                align="center",
            )
        ],
        width=800,
        height=400,
    )

    fig = go.Figure(data=[trace_pdf, trace_fill], layout=layout)

    return fig.show(fig_type)


def plot_prediction_intervals(
    intervals, y_pred, y_test, fig_type=None, width=800, height=400
):
    """
    Generates an interactive plot to visualize prediction intervals,
    actual values, and model predictions.

    Parameters:
    - intervals (numpy.ndarray): Array containing the lower and upper bounds of the prediction intervals.
    - y_pred (numpy.ndarray): Array containing the predicted values from the model.
    - y_test (pandas.Series): Series containing the actual test set values.

    Returns:
    - fig (plotly.graph_objects.Figure): Interactive Plotly figure object.
    """
    fig = go.Figure()

    lower_bound = intervals[:, 0]
    upper_bound = intervals[:, 1]

    # Valores reais
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode="lines",
            line=dict(color="darkblue"),
            name="Real Value",
        )
    )

    # Limite inferior
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=lower_bound,
            mode="lines",
            line=dict(color="rgba(128, 128, 128, 0.2)"),
            showlegend=False,
        )
    )

    # Limite superior
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=upper_bound,
            mode="lines",
            fill="tonexty",  # Preenche entre este trace e o anterior
            fillcolor="rgba(128, 128, 128, 0.2)",
            line=dict(color="rgba(128, 128, 128, 0.2)"),
            name="Prediction Intervals",
        )
    )

    # Previsões
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=y_pred,
            mode="lines",
            line=dict(color="red"),
            name="Prediction MidPoint",
        )
    )

    fig.update_layout(
        title="Interval Prediction",
        xaxis_title="Sample",
        yaxis_title="Value",
        legend=dict(title="Metric"),
        width=width,
        height=height,
    )

    return fig.show(fig_type)
