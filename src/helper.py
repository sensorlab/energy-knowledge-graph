import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_string(string: str) -> str:
    string = string.lower().strip()
    string = re.sub(' +', ' ', string)
    string = string.replace("_", " ")
    string = string.replace("-", " ")
    string = string.replace("&", " ")
    string = string.split("(")[0]
    string = string.split("#")[0]

    string = string.strip()

    # handle known synonyms
    synonyms = {
        "refrigerator": "fridge",
        "vaccumcleaner": "vacuum cleaner",
        "breadmaker": "bread maker",

    }
    if "freezer" in string:
        string = "fridge"

    if string in synonyms:
        string = synonyms[string]

    if 'hi fi' in string:
        string = "audio system"

    if "router" in string:
        string = "router"

    if "treadmill" in string:
        string = "running machine"

    if "laptop" in string:
        string = "laptop"

    if "server" in string:
        string = "server"

    if "monitor" in string and "baby" not in string:
        string = "monitor"
    # special cases
    if "computer" in string and "charger" not in string:
        string = "pc"

    if "tv" in string:
        string = "television"

    if "television" in string:
        string = "television"

    if "macbook" in string:
        string = "laptop"

    if "car charger" == string:
        string = "ev"

    if "toast" in string:
        string = "toaster"

    if "modem" in string:
        string = "router"

    # we treat all audio devices as speakers so subwoofer is also a speaker
    if "subwoofer" in string:
        string = "speaker"

    if "speaker" in string:
        string = "speaker"

    if "iron" in string and "soldering" not in string:
        string = "iron"

    if "coffeemachine" in string:
        string = "coffee machine"
    if "coffee maker" in string:
        string = "coffee machine"

    if "dishwasher" in string:
        string = "dish washer"
    if "air conditioner" in string:
        string = "ac"

    if "air conditioning" in string:
        string = "ac"

    string = re.sub(' +', ' ', string)
    string = re.sub(r'\d+', '', string)
    return string.strip()


# min-max normalization Xmin=0 
# noinspection PyPep8Naming
def normalize(X):
    max_value = 0

    for x in X:
        v = np.max(x)
        if v > max_value:
            max_value = v

    if max_value == 0:
        return X
    return X / max_value


# watts to kWh given data frequency as a fraction of an hour (e.g. 0.5 for half-hourly data)
def watts2kwh(df: pd.Series, data_frequency: float) -> pd.Series:
    df = df / 1000 * data_frequency
    return df


def generate_labels(data_path: Path, save_folder: Path, datasets: list[str]):
    """
    Generate labels for the given datasets and save to a pickle file in the save folder
    ### Parameters
    `data_path` : Path to the parsed data
    `save_folder` : Path to the save folder
    `datasets` : List of datasets to generate labels for, example: ["REFIT", "ECO"] will generate only for REFIT and ECO
    """
    print("Generating labels...")
    labels = set()
    for dataset in datasets:
        data = pd.read_pickle(data_path / dataset)
        for h in data:
            for k in data[h]:
                if "aggregate" in k:
                    continue
                labels.add(preprocess_string(k))

    # convert to list
    labels = list(labels)

    # save with pickle
    with open(save_folder / "labels.pkl", "wb") as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return labels


# save a dictionary to a pickle file
def save_to_pickle(dict: dict, filename: str):
    try:
        with open(filename, "wb") as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data successfully saved to ", filename)
    except Exception as e:
        print("Failed to save data: ", e)
