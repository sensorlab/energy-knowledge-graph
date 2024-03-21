import numpy as np
from pathlib import Path
import pandas as pd
import model_config as config
from sklearn.model_selection import train_test_split
from src.helper import normalize
from src.models.InceptionTime import Classifier_INCEPTION
import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, model will train on CPU.")


def read_data(data_path: str, label_path: str) -> tuple[np.ndarray, np.ndarray, np.array]:
    # resolve paths
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    label_path: Path = Path(label_path).resolve()
    assert label_path.exists(), f"Path '{label_path}' does not exist!"

    data = pd.read_pickle(data_path)
    labels = np.array(pd.read_pickle(label_path))

    X = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])

    y = y.astype(int)

    return X, y, labels


def train():
    base = Path(f"{config.SAVE_PATH}/{config.MODEL_NAME}")
    base.mkdir(parents=True, exist_ok=True)

    X, y, labels = read_data(config.DATA_PATH, config.LABELS_PATH)  # set paths

    # save config to json
    config.save_config()

    NmDevices = len(labels)  # number of devices(classes)

    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE,
                                                        random_state=config.RANDOM_STATE)

    # normalize data
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # create directory for data splits
    if not os.path.exists(base / "data_split/"):
        os.mkdir(base / "data_split/")

    # store data splits for reproducibility
    np.save(base / "data_split" / "X_train.npy", X_train)
    np.save(base / "data_split" / "X_test.npy", X_test)
    np.save(base / "data_split" / "y_train.npy", y_train)
    np.save(base / "data_split" / "y_test.npy", y_test)

    # create directory for model
    if not os.path.exists(base / "model/"):
        os.mkdir(base / "model/")

    # create directory for csv training logs
    if not os.path.exists(base / "logs/"):
        os.mkdir(base / "logs/")

    # train model
    for i in range(config.NUM_MODELS):
        # create csv logger callback to log training progress
        csv_logger = tf.keras.callbacks.CSVLogger(base / "logs" / f"model_{i}.csv", separator=",", append=False)
        model = Classifier_INCEPTION(
            output_directory=base / f"model/",
            input_shape=(config.WINDOW_SIZE, 1),
            nb_classes=NmDevices,
            verbose=True,
            build=True,
            batch_size=config.BATCH_SIZE,
            nb_epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            depth=config.DEPTH,
            model_number=i,
            kernel_size=config.KERNEL_SIZE
        )
        model.fit(X_train, y_train, csv_logger, validation=(X_test, y_test))


if __name__ == "__main__":
    train()
