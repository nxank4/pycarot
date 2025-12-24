import uuid

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import pycarot.anomaly
import pycarot.datasets


@pytest.fixture(scope="module")
def data():
    return pycarot.datasets.get_data("anomaly")


def test_anomaly(data):
    experiment_name = uuid.uuid4().hex
    pycarot.anomaly.setup(
        data,
        normalize=True,
        log_experiment=True,
        experiment_name=experiment_name,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create model
    iforest = pycarot.anomaly.create_model("iforest", experiment_custom_tags={"tag": 1})
    knn = pycarot.anomaly.create_model("knn", experiment_custom_tags={"tag": 1})
    # https://github.com/pycaret/pycaret/issues/3606
    cluster = pycarot.anomaly.create_model("cluster", experiment_custom_tags={"tag": 1})

    # Plot model
    pycarot.anomaly.plot_model(iforest)
    pycarot.anomaly.plot_model(knn)

    # assign model
    iforest_results = pycarot.anomaly.assign_model(iforest)
    knn_results = pycarot.anomaly.assign_model(knn)
    cluster_results = pycarot.anomaly.assign_model(cluster)
    assert isinstance(iforest_results, pd.DataFrame)
    assert isinstance(knn_results, pd.DataFrame)
    assert isinstance(cluster_results, pd.DataFrame)

    # predict model
    iforest_predictions = pycarot.anomaly.predict_model(model=iforest, data=data)
    knn_predictions = pycarot.anomaly.predict_model(model=knn, data=data)
    cluster_predictions = pycarot.anomaly.predict_model(model=cluster, data=data)
    assert isinstance(iforest_predictions, pd.DataFrame)
    assert isinstance(knn_predictions, pd.DataFrame)
    assert isinstance(cluster_predictions, pd.DataFrame)

    # get config
    X = pycarot.anomaly.get_config("X")
    seed = pycarot.anomaly.get_config("seed")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycarot.anomaly.set_config("seed", 124)
    seed = pycarot.anomaly.get_config("seed")
    assert seed == 124

    # returns table of models
    all_models = pycarot.anomaly.models()
    assert isinstance(all_models, pd.DataFrame)

    # Assert the custom tags are created
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for experiment_run in client.search_runs(experiment.experiment_id):
        run = client.get_run(experiment_run.info.run_id)
        assert run.data.tags.get("tag") == "1"

    # save model
    pycarot.anomaly.save_model(knn, "knn_model_23122019")

    # reset
    pycarot.anomaly.set_current_experiment(pycarot.anomaly.AnomalyExperiment())

    # load model
    knn = pycarot.anomaly.load_model("knn_model_23122019")

    # predict model
    knn_predictions = pycarot.anomaly.predict_model(model=knn, data=data)
    assert isinstance(knn_predictions, pd.DataFrame)


if __name__ == "__main__":
    test_anomaly()
