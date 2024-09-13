import os
import pandas as pd
from pathlib import Path
from src.helper import save_to_pickle


######################DATASET INFO#########################################
# sampling rate: 6s
# length: 1.5 years
# unit: watts
# households: 4
# no submeter appliance data
# Location: Poland
# Source: https://zenodo.org/records/5608475


# read file & set date as index 
def process_file(path: Path) -> pd.DataFrame:
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
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"
    data_dict = {}
    for house in os.listdir(data_path):
        if house.endswith(".csv"):
            df = process_file(data_path / house)
            data = {}
            for col in df.columns:
                if "AGGREGATE" in col:
                    data[col.lower()] = pd.DataFrame(df[col])
            data_dict[parse_name(house)] = data

    save_to_pickle(data_dict, save_path)
