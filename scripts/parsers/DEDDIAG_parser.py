import os
import pandas as pd
from helper_functions import *

def parse_id(file_name : str) -> int:
    return int(file_name.split('_')[1])


def parse_DEDDIAG(data_path : str, save_path: str):
    data = {}

    # extend data path
    data_path = os.path.join(data_path, "house_08/")
    # get map of item_id to label for appliance
    labels = pd.read_csv(data_path+"items.tsv", sep="\t")
    labels.set_index("item_id", inplace=True)
    id_label_map = labels["category"].to_dict()



    for device in ([d for d in os.listdir(data_path) if "data" in d]):
        label = id_label_map[parse_id(device)]
        # only get data for appliances
        if "Phase" not in label:
            # rename total to aggregate
            if "Total" in label:
                label = "aggregate"
            # preprocess data frame
            df = pd.read_csv(data_path + device, sep="\t")
            df["time"] = pd.to_datetime(df["time"])
            df.drop(columns=["item_id"], inplace=True)
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            df = df.resample("1s").ffill()
            df.dropna(inplace=True)
            data[label] = df

        
    data_dict = {
        "DEDDIAG_8": data,
    }

    save_to_pickle(data_dict, save_path)
