import pandas as pd

import pycarot.classification
import pycarot.datasets
from pycarot.internal.meta_estimators import CustomProbabilityThresholdClassifier


def test_probability_threshold():
    # loading dataset
    data = pycarot.datasets.get_data("juice")
    assert isinstance(data, pd.DataFrame)

    # init setup
    pycarot.classification.setup(
        data,
        target="Purchase",
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )

    probability_threshold = 0.75

    # compare models
    top3 = pycarot.classification.compare_models(
        n_select=100, exclude=["catboost"], probability_threshold=probability_threshold
    )[:3]
    assert isinstance(top3, list)
    assert isinstance(top3[0], CustomProbabilityThresholdClassifier)
    assert top3[0].probability_threshold == probability_threshold

    # tune model
    tuned_top3 = [pycarot.classification.tune_model(i, n_iter=3) for i in top3]
    assert isinstance(tuned_top3, list)
    assert isinstance(tuned_top3[0], CustomProbabilityThresholdClassifier)
    assert tuned_top3[0].probability_threshold == probability_threshold

    # ensemble model
    bagged_top3 = [
        pycarot.classification.ensemble_model(i, probability_threshold=probability_threshold)
        for i in tuned_top3
    ]
    assert isinstance(bagged_top3, list)
    assert isinstance(bagged_top3[0], CustomProbabilityThresholdClassifier)
    assert bagged_top3[0].probability_threshold == probability_threshold

    # blend models
    blender = pycarot.classification.blend_models(top3, probability_threshold=probability_threshold)
    assert isinstance(blender, CustomProbabilityThresholdClassifier)
    assert blender.probability_threshold == probability_threshold

    # stack models
    stacker = pycarot.classification.stack_models(
        estimator_list=top3[1:],
        meta_model=top3[0],
        probability_threshold=probability_threshold,
    )
    assert isinstance(stacker, CustomProbabilityThresholdClassifier)
    assert stacker.probability_threshold == probability_threshold

    # calibrate model
    calibrated = pycarot.classification.calibrate_model(estimator=top3[0])
    assert isinstance(calibrated, CustomProbabilityThresholdClassifier)
    assert calibrated.probability_threshold == probability_threshold

    # plot model
    lr = pycarot.classification.create_model("lr", probability_threshold=probability_threshold)
    pycarot.classification.plot_model(
        lr, save=True
    )  # scale removed because build failed due to large image size

    # select best model
    best = pycarot.classification.automl()
    assert isinstance(calibrated, CustomProbabilityThresholdClassifier)
    assert calibrated.probability_threshold == probability_threshold

    # hold out predictions
    predict_holdout = pycarot.classification.predict_model(lr)
    predict_holdout_0_5 = pycarot.classification.predict_model(lr, probability_threshold=0.5)
    predict_holdout_0_75 = pycarot.classification.predict_model(
        lr, probability_threshold=probability_threshold
    )
    assert isinstance(predict_holdout, pd.DataFrame)
    assert predict_holdout.equals(predict_holdout_0_75)
    assert not predict_holdout.equals(predict_holdout_0_5)

    # predictions on new dataset
    predict_holdout = pycarot.classification.predict_model(lr, data=data.drop("Purchase", axis=1))
    predict_holdout_0_5 = pycarot.classification.predict_model(
        lr, data=data.drop("Purchase", axis=1), probability_threshold=0.5
    )
    predict_holdout_0_75 = pycarot.classification.predict_model(
        lr,
        data=data.drop("Purchase", axis=1),
        probability_threshold=probability_threshold,
    )
    assert isinstance(predict_holdout, pd.DataFrame)
    assert predict_holdout.equals(predict_holdout_0_75)
    assert not predict_holdout.equals(predict_holdout_0_5)

    # finalize model
    final_best = pycarot.classification.finalize_model(best)
    assert isinstance(final_best._final_estimator, CustomProbabilityThresholdClassifier)
    assert final_best._final_estimator.probability_threshold == probability_threshold

    # save model
    pycarot.classification.save_model(best, "best_model_23122019")

    # load model
    saved_best = pycarot.classification.load_model("best_model_23122019")
    assert isinstance(saved_best._final_estimator, CustomProbabilityThresholdClassifier)
    assert saved_best._final_estimator.probability_threshold == probability_threshold

    assert 1 == 1


if __name__ == "__main__":
    test_probability_threshold()
