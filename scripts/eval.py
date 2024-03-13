import pandas as pd
import model_config
import numpy as np
from pathlib import Path
import os
import tensorflow as tf
from sklearn import metrics


def eval():

    # load test data
    model_path : Path = Path(config.MODEL_PATH + config.MODEL_NAME) 
    # the train.py script saves the data splits in the model folder the training data is already normalized
    X_test = np.load(model_path / "data_split/X_test.npy")
    y_test = np.load(model_path / "data_split/y_test.npy")
    labels = pd.read_pickle(config.LABELS_PATH)

    # load models
    models = []
    for f in os.listdir(model_path / "model"):
        # skip init file and jupyter notebook checkpoints
        if "init" in f or "ipynb" in f:
            continue

            
        model = tf.keras.models.load_model(model_path / "model"/f)

        models.append(model)

    # create predictions folder
    if not os.path.exists(model_path / "predictions"):
        os.mkdir(model_path / "predictions")

    # create results folder
    if not os.path.exists(model_path / "results"):
        os.mkdir(model_path / "results")

    model_predictions = []
    # predict
    for i,m in enumerate(models):
        y_pred = m.predict(X_test)
        # save individual predictions
        np.save(model_path / "predictions" / f"y_pred_{i}.npy", y_pred)

        # save classification report for each model
        r = metrics.classification_report(y_test, np.where(y_pred > config.THRESHOLD, 1, 0), target_names=labels, zero_division=0, output_dict=True)
        r_df = pd.DataFrame(r).T
        r_df.to_csv(model_path / "results" / f"classification_report_{i}.csv")
        model_predictions.append(y_pred)

    # average predictions from all models
    y_pred_avg = np.mean(model_predictions, axis=0)

    # save averaged predictions
    np.save(model_path / "predictions" / "y_pred_ensemble.npy", y_pred_avg)

    # apply threshold
    y_pred_tf = np.where(y_pred_avg > config.THRESHOLD, 1, 0)

    # ensemble classification report
    res = metrics.classification_report(y_test, y_pred_tf, target_names=labels, zero_division=0, output_dict=True)
    res_df = pd.DataFrame(res).T
    res_df.to_csv(model_path / "results" / "classification_report_ensemble.csv")

    

if __name__ == "__main__":
    eval()
        
