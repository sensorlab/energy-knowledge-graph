import pandas as pd
import os
import yaml


from helper_functions import save_to_pickle

######################DATASET INFO#########################################
# sampling rate: 6s
# unit: watts
# households: 5
# submetered: yes
# Location: United Kingdom
# Source: https://jack-kelly.com/data/

# gets the number of the device from the filename
def getNumber(device: str) -> int:
    return int((device.split(".")[0]).split("_")[1])


# preproces the file by reading it, converting the timestamp to datetime and setting it as index
def preproces_file(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None, sep=" ")
    df[0] = pd.to_datetime(df[0], unit="s")
    df = df.set_index(0)

    return df


# processes a house by reading all the files in the house folder and saving dfs for each device and returning them as a dictionary
def process_house(house_path: str, meta_path: str) -> dict:
    data = {}
    # read the labels file to get the names of the columns
    with open(meta_path) as file:
        meta = yaml.load(file, Loader=yaml.FullLoader)

    # create a dictionary to map the original names to the appliance types
    original_name_to_type = {}
    for appliance in meta["appliances"]:
        if "original_name" not in appliance:
            continue
        original_name = appliance["original_name"]
        appliance_type = appliance["type"]
        original_name_to_type[original_name] = appliance_type

    lables = pd.read_csv(house_path + "labels.dat", header=None, sep=" ")[1].values
    for device in os.listdir(house_path):
        if device.endswith(".dat"):
            if "channel" in device and "button" not in device:
                # get the number of the device from the filename
                number = getNumber(device)
                if lables[number - 1] in original_name_to_type:
                    name = original_name_to_type[lables[number - 1]]
                else:
                    name = lables[number - 1]
                df = preproces_file(house_path + device)
                data[name] = df

    return data


def parse_UKDALE(data_path: str, save_path: str) -> None:
    # dictionary of houses, each house is a dictionary of devices
    houses_data = {}
    for house in os.listdir(data_path):
        if "house" in house:
            number = house.split("_")[1]
            name = "UKDALE_" + number
            # skip due to lacking device submeter data(devices grouped together) and in general only 5 submeters
            if name == "UKDALE_4":
                continue
            houses_data[name] = process_house(data_path + "/" + house + "/", data_path + f"metadata/building{number}.yaml")

    save_to_pickle(houses_data, save_path)
