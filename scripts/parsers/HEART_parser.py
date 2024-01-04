import pandas as pd
import os
from helper_functions import *

######################DATASET INFO#########################################
# sampling rate: 1s
# unit: watts
# households: 4
# submetered: yes
# Location: Greece
# Source: https://zenodo.org/records/7997198
def parse_name(file_name: str):
    """
    Parse the file name to get the house name
    """
    # appliance name
    appliance_name = file_name.split(".")[0]

    # date
    return "HEART" + "_" + appliance_name[5:]


def parse_HEART(data_path : str, save_path : str):
    data_dict = {}
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            # 
            df = pd.read_csv(data_path + file)
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


    #save to pickle
    save_to_pickle(data_dict, save_path)

