import pandas as pd
from numpy import ndarray
from typing import Union, Dict, List
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_evaluate_classifier(clf: str, x_train: Union[List, ndarray], y_train: pd.DataFrame,
                              x_test: Union[List, ndarray],
                              y_test: pd.DataFrame) -> Dict[str, float]:

    """
    Train sklearn classifiers and evaluate on test data
    :param clf: Classifier to train. Currently one of{[svm, naive_bayes, random_forest, logistic_regression}
    :param x_train: Array/list containing train data for training
    :param y_train: Pandas column containing train target labels
    :param x_test: Array/list containing test data for evaluating the trained classifier
    :param y_test: Pandas column containing test target labels
    :return: Dictionary with accuracy/f1-scores on the test data
    """
    svm_cv_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'gamma': ('scale', 'auto')}
    scoring = {'Accuracy': 'accuracy', 'F-Measure': 'f1'}

    _classifier = None

    if clf == "svm":
        svc = SVC()
        _classifier = GridSearchCV(svc, svm_cv_parameters, n_jobs=6, scoring=scoring, verbose=3, refit="F-Measure")
    elif clf == "naive_bayes":
        _classifier = ComplementNB()
    elif clf == "random_forest":
        _classifier = RandomForestClassifier(n_estimators=100, n_jobs=6, verbose=1)
    elif clf == "logistic_regression":
        _classifier = LogisticRegression(n_jobs=6)

    _classifier.fit(x_train, y_train)
    acc = _classifier.score(x_test, y_test)
    f1 = f1_score(y_test, _classifier.predict(x_test))

    score_dict = {"acc": acc, "f1-score": f1}
    return score_dict
