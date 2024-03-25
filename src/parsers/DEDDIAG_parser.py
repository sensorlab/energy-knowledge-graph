import os
import pandas as pd
from pathlib import Path
from src.helper import save_to_pickle

######################DATASET INFO#########################################
# sampling rate: 1s
# length: 3.5 years
# unit: watts
# households: 15
# Only on household 8
# Location: Germany
# Source: https://www.nature.com/articles/s41597-021-00963-2


def parse_id(file_name: str) -> int:
    """
    Parse the dataset id from the file name
    ## Parameters
    file_name : The name of the file
    ## Returns
    int : The id
    """
    return int(file_name.split("_")[1])


def parse_DEDDIAG(data_path: str, save_path: str) -> None:
    """
    Parse the DEDDIAG dataset and save to a pickle file
    ## Parameters
    data_path : The path to the DEDDIAG dataset
    save_path : The path to save the parsed data in pickle format
    """
    data = {}

    # extend data path
    data_path: Path = Path(data_path).resolve() / "house_08"
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    # get map of item_id to label for appliance
    labels = pd.read_csv(data_path / "items.tsv", sep="\t")
    labels.set_index("item_id", inplace=True)
    id_label_map = labels["category"].to_dict()

    devices = [d for d in os.listdir(data_path) if "data" in d]

    for device in devices:
        label = id_label_map[parse_id(device)]
        # only get data for appliances
        if "Phase" not in label:
            # rename total to aggregate
            if "Total" in label:
                label = "aggregate"
            # preprocess data frame
            df = pd.read_csv(data_path / device, sep="\t")
            df["time"] = pd.to_datetime(df["time"])
            df.drop(columns=["item_id"], inplace=True)
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep="first")]
            df = df.resample("1s").ffill()
            df.dropna(inplace=True)
            data[label] = df

    data_dict = {"DEDDIAG_8": data}

    save_to_pickle(data_dict, save_path)
