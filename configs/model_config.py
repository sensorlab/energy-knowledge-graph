import json
import pandas as pd
MODEL_NAME = "Appliance_classification"  # used for saving model and predictions
# model parameters
NUM_EPOCHS = 1200
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEPTH = 10
KERNEL_SIZE = 128
NUM_MODELS = 10  # number of models in ensemble


# used for generating training data and labeling unlabeled data
WINDOW_SIZE = 2688
NUM_WINDOWS=100000
SAMPLING_RATE = "8S" 
UPPER_BOUND = pd.to_timedelta(32, unit='s') # if there is more than 15 gaps of this size in the window it will be discarded
MAX_GAP = pd.to_timedelta(3600, unit='s') # if there is a gap larger than this in the window it will be discarded

# paths to the data and labels used for training / evaluation
TRAINING_DATA_PATH = "./data/training_data/X_Y_wsize2688_numW_100000_upper32_gap3600_numD64.pkl"
# path to the labels generated by generate_training_data.py and the lables that will be used by the model
LABELS_PATH = "./data/training_data/labels_new.pkl"
# path to save the data/models
SAVE_PATH = "../data/"
MODEL_PATH = "./data/trained_models/"

# folder to save the generated training data
TRAINING_DATA_FOLDER = "./data/training_data/"




USE_SEED = True
RANDOM_STATE = 42
SAVE_PREDICTIONS = True # save predictions for each model
SAVE_MODEL = True # save the models
THRESHOLD = 0.3 # threshold for classification
TEST_SIZE = 0.2 # size of the test set


# datasets used for training
TRAINING_DATASETS = [
    "DEDDIAG",
    "DRED",
    "ECO",
    "ENERTALK",
    "HEART",
    "HES",
    "IAWE",
    "REDD",
    "REFIT",
    "UKDALE"
]


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
        "DATA_PATH": TRAINING_DATA_PATH,
        "SAVE_PATH": SAVE_PATH,
        "LABELS_PATH": LABELS_PATH,
        "LABELS_ARRAY_BACKUP": pd.read_pickle(LABELS_PATH),
        "USE_SEED": USE_SEED,
        "RANDOM_STATE": RANDOM_STATE,
        "SAVE_PREDICTIONS": SAVE_PREDICTIONS,
        "SAVE_MODEL": SAVE_MODEL,
        "THRESHOLD": THRESHOLD,
        "TEST_SIZE": TEST_SIZE,
    }

    with open(f"{SAVE_PATH}/{MODEL_NAME}/config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
