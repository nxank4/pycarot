import sys
import uuid

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import pycarot.clustering
import pycarot.datasets


if sys.platform == "win32":
    pytest.skip("Skipping test module on Windows", allow_module_level=True)


@pytest.fixture(scope="module")
def data():
    return pycarot.datasets.get_data("jewellery")


def test_clustering(data):
    experiment_name = uuid.uuid4().hex
    pycarot.clustering.setup(
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
    kmeans = pycarot.clustering.create_model("kmeans", experiment_custom_tags={"tag": 1})
    kmodes = pycarot.clustering.create_model("kmodes", experiment_custom_tags={"tag": 1})

    # Plot Model
    pycarot.clustering.plot_model(kmeans)
    pycarot.clustering.plot_model(kmodes)

    # assign model
    kmeans_results = pycarot.clustering.assign_model(kmeans)
    kmodes_results = pycarot.clustering.assign_model(kmodes)
    assert isinstance(kmeans_results, pd.DataFrame)
    assert isinstance(kmodes_results, pd.DataFrame)

    # predict model
    kmeans_predictions = pycarot.clustering.predict_model(model=kmeans, data=data)
    assert isinstance(kmeans_predictions, pd.DataFrame)

    # returns table of models
    all_models = pycarot.clustering.models()
    assert isinstance(all_models, pd.DataFrame)

    # get config
    X = pycarot.clustering.get_config("X")
    seed = pycarot.clustering.get_config("seed")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(seed, int)

    # set config
    pycarot.clustering.set_config("seed", 124)
    seed = pycarot.clustering.get_config("seed")
    assert seed == 124

    # Assert the custom tags are created
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for experiment_run in client.search_runs(experiment.experiment_id):
        run = client.get_run(experiment_run.info.run_id)
        assert run.data.tags.get("tag") == "1"

    # save model
    pycarot.clustering.save_model(kmeans, "kmeans_model_23122019")

    # reset
    pycarot.clustering.set_current_experiment(pycarot.clustering.ClusteringExperiment())

    # load model
    kmeans = pycarot.clustering.load_model("kmeans_model_23122019")

    # predict model
    kmeans_predictions = pycarot.clustering.predict_model(model=kmeans, data=data)
    assert isinstance(kmeans_predictions, pd.DataFrame)


if __name__ == "__main__":
    test_clustering()
