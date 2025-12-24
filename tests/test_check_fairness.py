import pandas as pd

import pycarot.classification
import pycarot.datasets
import pycarot.regression


def test_check_fairness_binary_classification():
    # loading dataset
    data = pycarot.datasets.get_data("income")

    # initialize setup
    pycarot.classification.setup(
        data,
        target="income >50K",
        html=False,
        n_jobs=1,
    )

    # train model
    lightgbm = pycarot.classification.create_model("lightgbm", fold=3)

    # check fairness
    lightgbm_fairness = pycarot.classification.check_fairness(lightgbm, ["sex"])
    assert isinstance(lightgbm_fairness, pd.DataFrame)


def test_check_fairness_multiclass_classification():
    # loading dataset
    data = pycarot.datasets.get_data("iris")

    # initialize setup
    pycarot.classification.setup(
        data,
        target="species",
        html=False,
        n_jobs=1,
        train_size=0.8,
    )

    # train model
    lightgbm = pycarot.classification.create_model("lightgbm", cross_validation=False)

    # check fairness
    lightgbm_fairness = pycarot.classification.check_fairness(lightgbm, ["sepal_length"])
    assert isinstance(lightgbm_fairness, pd.DataFrame)


def test_check_fairness_regression():
    # loading dataset
    data = pycarot.datasets.get_data("boston")

    # initialize setup
    pycarot.regression.setup(
        data,
        target="medv",
        html=False,
        n_jobs=1,
    )

    # train model
    lightgbm = pycarot.regression.create_model("lightgbm", fold=3)

    # check fairness
    lightgbm_fairness = pycarot.regression.check_fairness(lightgbm, ["chas"])
    assert isinstance(lightgbm_fairness, pd.DataFrame)


if __name__ == "__main__":
    test_check_fairness_binary_classification()
    test_check_fairness_multiclass_classification()
    test_check_fairness_regression()
