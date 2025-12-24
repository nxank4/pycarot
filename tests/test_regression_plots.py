import pandas as pd
import pytest

import pycarot.datasets
import pycarot.regression


@pytest.mark.plotting
def test_plot():
    # loading dataset
    data = pycarot.datasets.get_data("boston")
    assert isinstance(data, pd.DataFrame)

    # init setup
    pycarot.regression.setup(
        data,
        target="medv",
        log_experiment=True,
        log_plots=True,
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )

    model = pycarot.regression.create_model("rf", max_depth=2, n_estimators=5)

    exp = pycarot.regression.RegressionExperiment()
    available_plots = exp._available_plots

    for plot in available_plots:
        pycarot.regression.plot_model(model, plot=plot)

    models = [
        pycarot.regression.create_model("et"),
        pycarot.regression.create_model("xgboost"),
    ]

    # no pfi due to dependency hell
    available_shap = ["summary", "correlation", "reason", "pdp", "msa"]

    for model in models:
        for plot in available_shap:
            pycarot.regression.interpret_model(model, plot=plot)
            pycarot.regression.interpret_model(
                model, plot=plot, X_new_sample=data.drop("medv", axis=1).iloc[:10]
            )

    assert 1 == 1


if __name__ == "__main__":
    test_plot()
