import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# min-max normalization Xmin=0 
def normalize(X, max_value):
    """
    Normalize the data
    ## Parameters
    `X` : Numpy array containing the data
    `max_value` : max value in the data
    ## Returns
    `X` : Normalized data
    """

    for x in X:
        v = np.max(x)
        if v > max_value:
            max_value = v

    if max_value == 0:
        return X
    return X / max_value


# noinspection PyUnboundLocalVariable
def process_data(df: pd.DataFrame, time_window: int, upper_bound: pd.Timedelta, max_gap: pd.Timedelta) -> list:
    """
    Process the data by resampling it to 8s and filling the gaps with the nearest value and then splitting it into windows of size time_window.
    If there is a gap of more than max_gap skip the window. If there are more than 15 gaps of upper_bound or more skip the window. If the device is always off skip the window.
    ## Parameters
    `df` : DataFrame containing the data
    `time_window` : size of the window in rows 
    `upper_bound` : upper bound for the gap in seconds if there is more than 15 gaps of this size in a window skip the window
    `max_gap` : max gap in seconds if there is a gap of more than this size in a window skip the window
    ## Returns
    `windows` : List of windows for the aggregated data
    """
    df = df.resample("8S").fillna(method="nearest", limit=4)
    df.fillna(0, inplace=True)
    # handle negatve values
    df[df < 0] = 0
    windows = []
    for i in range(0, len(df) - time_window, time_window + 1):
        window = df.iloc[i: i + time_window]
        # if there is a gap of more than max_gap skip the window
        time_diffs = window.index.to_series().diff().dropna()
        if (time_diffs >= max_gap).any():
            continue
        # if there are more than 15 gaps of upper_bound or more skip the window
        if len(time_diffs[time_diffs > upper_bound]) > 15:
            continue
        # skip if the device is always off
        if window.max().max() < 5:
            continue
        window.reset_index(drop=True, inplace=True)

        window_values = window.values
        max_value = np.max(window_values)

        windows.append(window_values)

    return np.array(windows), max_value


def preprocess_dataset(data_path: Path):
    """
    Preprocess the datasets by resampling the data and splitting it into windows
    ## Parameters
    `data_path` : Path to the dataset
    ## Returns
    `household_windows` : Dictionary containing the windows for each household
    `max_value` : max value in the data used for normalization
    """
    max_value = 0
    household_windows = {}
    data = pd.read_pickle(data_path)
    for h in tqdm(data):
        windows, max_value_window = process_data(data[h]["aggregate"], 2688, pd.Timedelta("32s"), pd.Timedelta("3600s"))

        if max_value_window > max_value:
            max_value = max_value_window

        household_windows[h] = windows

    return household_windows, max_value


def predict_appilances(windows: np.array, models: list, max_value: float) -> dict:
    """
    Predict the appliances for the given windows using the models
    ## Parameters
    `windows` : Numpy array containing the windows
    `models` : List of models to use for prediction
    `max_value` : max value in the data used for normalization
    ## Returns
    `y_pred_tf` : vector containing the predictions
    """
    predictions = []
    windows = normalize(windows, max_value)
    # predict for each model
    for model in models:
        y_pred = model.predict(windows)

        predictions.append(y_pred)

    # average the predictions of the models
    predictions_models = np.array(predictions)
    predictions_models = np.mean(predictions_models, axis=0)

    predictions_houses = np.mean(predictions_models, axis=0)

    # threshold the predictions
    y_pred_tf = np.where(predictions_houses > 0.3, 1, 0)

    return y_pred_tf


def get_labels(data: dict, model_path: Path, label_path: Path, max_value: float) -> dict:
    """
    Get the appliances for the given households
    ## Parameters
    `data` : Dictionary containing the windows for each household
    `model_path` : Path to the folder containing the models
    `label_path` : Path to the labels
    `max_value` : max value in the data used for normalization
    ## Returns
    `devices` : Dictionary containing the predicted appliances for each household
    """
    # load labels	
    labels = np.array(pd.read_pickle(label_path))
    # load models
    models = []
    for f in os.listdir(model_path / "model"):
        # skip init file and jupyter notebook checkpoints
        if "init" in f or "ipynb" in f:
            continue

        model = tf.keras.models.load_model(model_path / "model" / f)

        models.append(model)

    devices = {}
    for house in data:
        devices[house] = labels[predict_appilances(data[house], models, max_value) == 1]
    return devices


def get_predicted_appliances(data_path: Path, model_path: Path, label_path: Path, save_path: Path,
                             datasets: list[str]) -> None:
    """
    Label unlabeled datasets utilizing InceptionTime model.
    ## Parameters
    `data_path` : Path to the parsed data
    `model_path` : Path to the pretrained model folder
    `label_path` : Path to the labels
    `save_path` : Path to the save folder to save the predicted devices
    `datasets` : List of datasets to generate labels for, example: 'IDEAL' will generate only for IDEAL


    """
    household_labels = {}
    for dataset in datasets:
        assert (data_path / (dataset + ".pkl")).exists(), f"Dataset {dataset} does not exist"

        household_windows, max_value = preprocess_dataset(data_path / (dataset + ".pkl"))

        labels = get_labels(household_windows, model_path, label_path, max_value)
        household_labels.update(labels)

    # save with pickle
    with open(Path(save_path) / "predicted_devices.pkl", "wb") as handle:
        pickle.dump(household_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label unlabeled datasets utilizing InceptionTime model.")
    parser.add_argument("--data_path", type=str, default="./data/parsed/", help="Path to the parsed data")
    parser.add_argument("--model_path", type=str, default="./data/trained_models/InceptionTime/",
                        help="Path to the pretrained model folder")
    parser.add_argument("--label_path", type=str, default="./data/training_data/labels.pkl", help="Path to the labels")
    parser.add_argument("--save_folder", type=str, default="./data/", help="Path to the save folder")
    parser.add_argument("--datasets", type=str, nargs="+", default=["IDEAL", "LERTA"],
                        help="List of datasets to generate labels for, example: 'IDEAL' will generate only for IDEAL")
    args = parser.parse_args()

    data_path: Path = Path(args.data_path).resolve()
    assert data_path.exists(), "Data path does not exist"

    model_path: Path = Path(args.model_path).resolve()
    assert model_path.exists(), "Model path does not exist"

    label_path: Path = Path(args.label_path).resolve()
    assert label_path.exists(), "Label path does not exist"

    save_path: Path = Path(args.save_folder).resolve()
    assert save_path.exists(), "Save path does not exist"

    datasets = args.datasets
    # check if the datasets exist

    household_labels = {}
    for dataset in datasets:
        assert (data_path / (dataset + ".pkl")).exists(), f"Dataset {dataset} does not exist"

        household_windows, max_value = preprocess_dataset(data_path / (dataset + ".pkl"))

        labels = get_labels(household_windows, model_path, label_path, max_value)
        household_labels.update(labels)

    # save with pickle
    with open(Path(save_path) / "predicted_devices.pkl", "wb") as handle:
        pickle.dump(household_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
