import pandas as pd
import os
from helper_functions import save_to_pickle
from pathlib import Path



######################DATASET INFO#########################################
# sampling rate: 1s
# length: 1 month
# unit: watts
# households: 4
# submetered: yes
# Location: Greece
# Source: https://zenodo.org/records/7997198

def parse_name(file_name: str) -> str:
    """
    Parse the file name to get the house name
    """
    # appliance name
    appliance_name = file_name.split(".")[0]

    # date
    return "HEART" + "_" + appliance_name[5:]


def parse_HEART(data_path: str, save_path: str) -> None:
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"
    data_dict = {}
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            #
            df = pd.read_csv(data_path / file)
            df.drop(columns=["router"], inplace=True)
            # convert unix timestamp to datetime
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Europe/Athens")
            # set datetime as index and drop unnecessary columns
            df = df.set_index("Timestamp").drop(columns=["dw", "wm"])
            df.sort_index(inplace=True)

            df.rename(columns={"Value": "aggregate"}, inplace=True)

            df.dropna(inplace=True)
            # create a dictionary of dataframes for each device
            devices_dict = {}
            for device in df.columns:
                devices_dict[device] = pd.DataFrame(df[device])

            # add the device dictionary to the data dictionary
            data_dict[parse_name(file)] = devices_dict

    # save to pickle
    save_to_pickle(data_dict, save_path)
