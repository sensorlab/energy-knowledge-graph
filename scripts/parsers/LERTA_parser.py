import os
import pandas as pd
from helper_functions import save_to_pickle


######################DATASET INFO#########################################
# sampling rate: 6s
# unit: watts
# households: 4
# no submeter appliance data
# Location: Poland
# Source: https://zenodo.org/records/5608475

# read file set date as index and convert to kWh
def process_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("Time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    return df


# parse house name from file name
def parse_name(file_name: str) -> str:
    name = file_name.split(".")[0]
    name = name.split("_")[-1]
    name = "LERTA_" + name[-1]
    return name


def parse_LERTA(data_path: str, save_path: str) -> None:
    houses_data = {}
    for house in os.listdir(data_path):
        if house.endswith(".csv"):
            df = process_file(data_path + house)
            data = {}
            for col in df.columns:
                if "AGGREGATE" in col:
                    data[col.lower()] = pd.DataFrame(df[col])
            houses_data[parse_name(house)] = data

    save_to_pickle(houses_data, save_path)
