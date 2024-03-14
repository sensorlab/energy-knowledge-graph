import json
import os
MODEL_NAME="TEST" # used for saving model and predictions
# model parameters
NUM_EPOCHS=1200
BATCH_SIZE=128
LEARNING_RATE=0.001
DEPTH=10
KERNEL_SIZE=128
NUM_MODELS=10 # number of models in ensemble
WINDOW_SIZE=2688


# paths
DATA_PATH=".data/training_data/X_Y_wsize2688_numW_100000_upper32_gap3600_numD64_ideal.pkl"
SAVE_PATH= "../data/"
LABELS_PATH=".data/training_data/labels_new.pkl"
MODEL_PATH="./trained_models/"


USE_SEED=True
RANDOM_STATE=42
SAVE_PREDICTIONS=True
SAVE_MODEL=True
THRESHOLD=0.3
TEST_SIZE=0.2


def save_config():
    # save config
    config_dict = {
        "MODEL_NAME": MODEL_NAME,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "DEPTH": DEPTH,
        "KERNEL_SIZE": KERNEL_SIZE,
        "NUM_MODELS": NUM_MODELS,
        "WINDOW_SIZE": WINDOW_SIZE,
        "DATA_PATH": DATA_PATH,
        "SAVE_PATH": SAVE_PATH,
        "LABELS_PATH": LABELS_PATH,
        "USE_SEED": USE_SEED,
        "RANDOM_STATE": RANDOM_STATE,
        "SAVE_PREDICTIONS": SAVE_PREDICTIONS,
        "SAVE_MODEL": SAVE_MODEL,
        "THRESHOLD": THRESHOLD,
        "TEST_SIZE": TEST_SIZE,
    }

    with open(f"{SAVE_PATH}/{MODEL_NAME}/config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
