import sys

import pytest

import pycarot.classification
import pycarot.datasets
import pycarot.regression


if sys.platform == "win32":
    pytest.skip("Skipping test module on Windows", allow_module_level=True)


def test_classification_create_docker():
    # loading dataset
    data = pycarot.datasets.get_data("blood")

    # initialize setup
    pycarot.classification.setup(
        data,
        target="Class",
        html=False,
        n_jobs=1,
    )

    # train model
    lr = pycarot.classification.create_model("lr")

    # create api
    pycarot.classification.create_api(lr, "blood_api")
    pycarot.classification.create_docker("blood_api")
    assert 1 == 1


def test_regression_create_docker():
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
    lr = pycarot.regression.create_model("lr")

    # create api
    pycarot.regression.create_api(lr, "boston_api")
    pycarot.regression.create_docker("boston_api")
    assert 1 == 1


if __name__ == "__main__":
    test_classification_create_docker()
    test_regression_create_docker()
