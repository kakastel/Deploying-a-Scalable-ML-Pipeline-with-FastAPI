import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics


def test_model_returns_random_forest():
    """
    Test checks train_model properly returns a fitted RandomForestClassifier.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])

    test_result_model = train_model(X_train, y_train)
    assert isinstance(test_result_model, RandomForestClassifier)


def test_array_returned():
    """
    Test to check that inference returns the appropriate array
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])

    test_result_model = train_model(X_train, y_train)
    test_array = inference(test_result_model, X_train)
    assert isinstance(test_array, np.ndarray)
    assert len(test_array) == len(X_train)


def test_metrics_with_known_values():
    """
    Testing that compute_model_metrics gives correct values for a known case.
    """
    y_true = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])

    precision, recall, f_beta = compute_model_metrics(y_true, preds)
    assert precision == 0.5
    assert recall == 0.5
    assert f_beta == 0.5
