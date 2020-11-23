"""
Example of a data science pipeline
"""

import io
import json

import pandas as pd
import pytest
import requests
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import dds
from .utils import cleandir

_ = cleandir

path_raw = "/wine-quality/raw"
path_model = "/wine-quality/my_model"
path_model_stats = "/wine-quality/my_model_stats.json"


def _load_data():
    url = "https://raw.githubusercontent.com/zygmuntz/wine-quality/master/winequality/winequality-red.csv"
    x = requests.get(url=url, verify=False).content
    return pd.read_csv(io.StringIO(x.decode("utf8")), sep=";")


def load_data():
    return dds.keep(path_raw, _load_data)


def build_model(X_train, y_train):
    pipeline_ = make_pipeline(
        preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100)
    )
    hyperparameters = {
        "randomforestregressor__max_features": ["log2"],
        "randomforestregressor__max_depth": [None, 1],
    }

    clf = GridSearchCV(pipeline_, hyperparameters, cv=2)
    clf.fit(X_train, y_train)
    return clf


def model_stats(clf, X_test, y_test):
    pred = clf.predict(X_test)
    return json.dumps(
        {"r2_score": r2_score(y_test, pred), "mse": mean_squared_error(y_test, pred)}
    )


def pipeline():
    wine_data = load_data()
    y = wine_data.quality
    X = wine_data.drop("quality", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=123, stratify=y
    )
    clf = dds.keep(path_model, build_model, X_train, y_train)
    dds.keep(path_model_stats, model_stats, clf, X_test, y_test)


# @pytest.mark.usefixtures("cleandir")
# def test_sklearn():
#     """ Unauthorized objects are not taken into account """
#     dds.eval(pipeline)
#     dds.eval(pipeline)
