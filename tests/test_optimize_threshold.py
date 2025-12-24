import pandas as pd

import pycarot.classification
import pycarot.datasets
from pycarot.internal.meta_estimators import CustomProbabilityThresholdClassifier


def test_optimize_threshold():
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

    # optimize threshold
    optimized_data, optimized_model = pycarot.classification.optimize_threshold(
        lr, return_data=True
    )
    assert isinstance(optimized_data, pd.core.frame.DataFrame)
    assert isinstance(optimized_model, CustomProbabilityThresholdClassifier)


if __name__ == "__main__":
    test_optimize_threshold()
