import pycarot.classification
import pycarot.datasets
import pycarot.regression


def test_classification_convert_model():
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

    # convert model
    lr_java = pycarot.classification.convert_model(lr, "java")
    assert isinstance(lr_java, str)


def test_regression_convert_model():
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

    # convert model
    lr_java = pycarot.regression.convert_model(lr, "java")
    assert isinstance(lr_java, str)


if __name__ == "__main__":
    test_classification_convert_model()
    test_regression_convert_model()
