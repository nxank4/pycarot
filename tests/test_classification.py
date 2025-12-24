import uuid

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.metrics import recall_score

import pycarot.classification
import pycarot.datasets


@pytest.fixture(scope="module")
def juice_dataframe():
    # loading dataset
    return pycarot.datasets.get_data("juice")


@pytest.mark.parametrize("return_train_score", [True, False])
def test_classification(juice_dataframe, return_train_score):
    assert isinstance(juice_dataframe, pd.core.frame.DataFrame)

    # init setup
    pycarot.classification.setup(
        juice_dataframe,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # compare models
    top3 = pycarot.classification.compare_models(errors="raise", n_select=100)[:3]
    assert isinstance(top3, list)
    metrics = pycarot.classification.pull()
    # no metric should be 0
    assert (
        (
            metrics.loc[[i for i in metrics.index if i not in ("dummy")]][
                [c for c in metrics.columns if c not in ("Model", "TT (Sec)")]
            ]
            != 0
        )
        .all()
        .all()
    )

    # tune model
    tuned_top3 = [
        pycarot.classification.tune_model(i, n_iter=3, return_train_score=return_train_score)
        for i in top3
    ]
    assert isinstance(tuned_top3, list)

    pycarot.classification.tune_model(
        top3[0], n_iter=3, choose_better=True, return_train_score=return_train_score
    )

    # ensemble model
    bagged_top3 = [
        pycarot.classification.ensemble_model(i, return_train_score=return_train_score)
        for i in tuned_top3
    ]
    assert isinstance(bagged_top3, list)

    # blend models
    pycarot.classification.blend_models(top3, return_train_score=return_train_score)

    # stack models
    stacker = pycarot.classification.stack_models(
        estimator_list=top3, return_train_score=return_train_score
    )
    pycarot.classification.predict_model(stacker)

    # plot model
    lr = pycarot.classification.create_model("lr", return_train_score=return_train_score)
    pycarot.classification.plot_model(lr, save=True, scale=5)

    # select best model
    pycarot.classification.automl(optimize="MCC", use_holdout=True)
    best = pycarot.classification.automl(optimize="MCC")

    # hold out predictions
    predict_holdout = pycarot.classification.predict_model(best)
    assert isinstance(predict_holdout, pd.DataFrame)

    # predictions on new dataset
    predict_holdout = pycarot.classification.predict_model(best, data=juice_dataframe)
    assert isinstance(predict_holdout, pd.DataFrame)

    # calibrate model
    pycarot.classification.calibrate_model(best, return_train_score=return_train_score)

    # finalize model
    pycarot.classification.finalize_model(best)

    # save model
    pycarot.classification.save_model(best, "best_model_23122019")

    # load model
    pycarot.classification.load_model("best_model_23122019")

    # returns table of models
    all_models = pycarot.classification.models()
    assert isinstance(all_models, pd.DataFrame)

    # get config
    X_train = pycarot.classification.get_config("X_train")
    X_test = pycarot.classification.get_config("X_test")
    y_train = pycarot.classification.get_config("y_train")
    y_test = pycarot.classification.get_config("y_test")
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # set config
    pycarot.classification.set_config("seed", 124)
    seed = pycarot.classification.get_config("seed")
    assert seed == 124

    assert 1 == 1


def test_classification_predict_on_unseen(juice_dataframe):
    exp = pycarot.classification.ClassificationExperiment()
    # init setup
    exp.setup(
        juice_dataframe,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )
    model = exp.create_model("dt", cross_validation=False)

    # save model
    exp.save_model(model, "best_model_23122019")

    exp = pycarot.classification.ClassificationExperiment()
    # load model
    model = exp.load_model("best_model_23122019")
    exp.predict_model(model, juice_dataframe)


def test_classification_custom_metric(juice_dataframe):
    exp = pycarot.classification.ClassificationExperiment()
    # init setup
    exp.setup(
        juice_dataframe,
        target="Purchase",
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    # create a custom function (sklearn >=1.3.0 requires kwargs in func def)
    def specificity(y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, pos_label=0, zero_division=1)

    # add metric to PyCaret
    exp.add_metric("specificity", "specificity", specificity, greater_is_better=True)

    lr = exp.create_model("lr")
    assert exp.pull()["specificity"].sum() != 0

    exp.predict_model(lr)
    assert exp.pull()["specificity"].sum() != 0


class TestClassificationExperimentCustomTags:
    def test_classification_setup_fails_with_experiment_custom_tags(self, juice_dataframe):
        with pytest.raises(Exception):
            # init setup
            _ = pycarot.classification.setup(
                juice_dataframe,
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags="custom_tag",
            )

    @pytest.mark.parametrize("custom_tag", [1, ("pytest", "True"), True, 1000.0])
    def test_classification_setup_fails_with_experiment_custom_multiples_inputs(self, custom_tag):
        with pytest.raises(Exception):
            # init setup
            _ = pycarot.classification.setup(
                pycarot.datasets.get_data("juice"),
                target="Purchase",
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                log_experiment=True,
                html=False,
                session_id=123,
                n_jobs=1,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag,
            )

    def test_classification_models_with_experiment_custom_tags(self, juice_dataframe):
        # init setup
        experiment_name = uuid.uuid4().hex
        _ = pycarot.classification.setup(
            juice_dataframe,
            target="Purchase",
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            log_experiment=True,
            html=False,
            session_id=123,
            n_jobs=1,
            experiment_name=experiment_name,
        )

        # compare models
        _ = pycarot.classification.compare_models(
            errors="raise", n_select=100, experiment_custom_tags={"pytest": "testing"}
        )[:3]

        # get experiment data
        tracking_api = MlflowClient()
        experiment = tracking_api.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        # get run's info
        experiment_run = tracking_api.search_runs(experiment_id)[0]
        # get run id
        run_id = experiment_run.info.run_id
        # get run data
        run_data = tracking_api.get_run(run_id)
        # assert that custom tag was inserted
        assert run_data.to_dictionary().get("data").get("tags").get("pytest") == "testing"


if __name__ == "__main__":
    test_classification()
    test_classification_predict_on_unseen()
    TestClassificationExperimentCustomTags()
