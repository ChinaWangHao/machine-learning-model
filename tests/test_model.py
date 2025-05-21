import pytest
from spam_detector.model import create_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def test_create_naive_bayes_model():
    config = {
        "model": {"type": "naive_bayes"},
        "naive_bayes": {"alpha": 0.5, "fit_prior": False},
        "data": {"random_state": 42}
    }
    model = create_model(config)
    assert isinstance(model, MultinomialNB)
    assert model.alpha == 0.5
    assert model.fit_prior == False


def test_create_svm_model():
    config = {
        "model": {"type": "svm"},
        "svm": {"C": 2.0, "kernel": "rbf", "gamma": "auto"},
        "data": {"random_state": 42}
    }
    model = create_model(config)
    assert isinstance(model, SVC)
    assert model.C == 2.0
    assert model.kernel == "rbf"
    assert model.gamma == "auto"


def test_create_random_forest_model():
    config = {
        "model": {"type": "random_forest"},
        "random_forest": {"n_estimators": 50, "max_depth": 10},
        "data": {"random_state": 42}
    }
    model = create_model(config)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 50
    assert model.max_depth == 10
    assert model.random_state == 42


def test_create_logistic_regression_model():
    config = {
        "model": {"type": "logistic_regression"},
        "logistic_regression": {"C": 0.5, "penalty": "l1", "solver": "saga"},
        "data": {"random_state": 42}
    }
    model = create_model(config)
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.5
    assert model.penalty == "l1"
    assert model.solver == "saga"
    assert model.random_state == 42


def test_invalid_model_type():
    config = {"model": {"type": "invalid_type"}}
    with pytest.raises(ValueError):
        create_model(config)
