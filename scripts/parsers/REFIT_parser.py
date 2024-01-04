import pandas as pd
from helper_functions import save_to_pickle
import os

######################DATASET INFO#########################################
# sampling rate: 8s
# unit: watts
# households: 20
# submetered
# Location: United Kingdom
# Source: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

# appliance names for each house
appliances = [
        'aggregate, fridge, chest freezer, upright freezer, tumble dryer, washing machine, dishwasher, computer site, television site, electric heater',
        'aggregate, fridge-freezer, washing machine, dishwasher, television, microwave, toaster, hi-fi, kettle, oven extractor fan',
        'aggregate, toaster, fridge-freezer, freezer, tumble dryer, dishwasher, washing machine, television, microwave, kettle',
        'aggregate, fridge, freezer, fridge-freezer, washing machine (1), washing machine (2), computer site, television site, microwave, kettle',
        'aggregate, fridge-freezer, tumble dryer 3, washing machine, dishwasher, computer site, television site, combination microwave, kettle, toaster',
        'aggregate, freezer (utility room), washing machine, dishwasher, mjy computer, television site, microwave, kettle, toaster, pgm computer',
        'aggregate, fridge, freezer (garage), freezer, tumble dryer, washing machine, dishwasher, television site, toaster, kettle',
        'aggregate, fridge, freezer, dryer, washing machine, toaster, computer, television site, microwave, kettle',
        'aggregate, fridge-freezer, washer dryer, washing machine, dishwasher, television site, microwave, kettle, hi-fi, electric heater',
        'aggregate, blender, freezer, chest freezer (in garage), fridge-freezer, washing machine, dishwasher, television site, microwave, food processor',
        'aggregate, fridge, fridge-freezer, washing machine, dishwasher, computer site, microwave, kettle, router, hi-fi',
        'aggregate, fridge-freezer, television site(lounge), microwave, kettle, toaster, television site (bedroom), not used, not used, not used',
        'aggregate, television site, unknown, washing machine, dishwasher, tumble dryer, television site2, computer site, microwave, kettle',
        None,
        'aggregate, fridge-freezer, tumble dryer, washing machine, dishwasher, computer site, television site, microwave, kettle, toaster',
        'aggregate, fridge-freezer (1), fridge-freezer (2), electric heater (1)?, electric heater (2), washing machine, dishwasher, computer site, television site, dehumidifier/heater',
        'aggregate, freezer (garage), fridge-freezer, tumble dryer (garage), washing machine, computer site, television site, microwave, kettle, plug site (bedroom)',
        'aggregate, fridge(garage), freezer(garage), fridge-freezer, washer dryer(garage), washing machine, dishwasher, desktop computer, television site, microwave',
        'aggregate, fridge & freezer, washing machine, television site, microwave, kettle, toaster, bread-maker, lamp (80watts), hi-fi',
        'aggregate, fridge, freezer, tumble dryer, washing machine, dishwasher, computer site, television site, microwave, kettle',
        'aggregate, fridge-freezer, tumble dryer, washing machine, dishwasher, food mixer, television, kettle/toaster, vivarium, pond pump',
]



def process_dataframe(df: pd.DataFrame, house_number) -> pd.DataFrame:
    df.drop(columns=["Unix", "Issues"], inplace=True)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time").sort_index()
    df = df.resample("8s").fillna(method="nearest", limit=1).dropna()


    device_names = appliances[house_number-1].split(",")
    if device_names != None:
        df.columns = device_names
    # dictionary to hold devices for each house
    data_dict = {}
    for c in df.columns:
        if "not used" in c or "unknown" in c:
                continue
        data_dict[c] = pd.DataFrame(df[c])
    
    return data_dict


def parse_name(name: str) -> int:
    name = name.split(".")[0]
    return int(name.split("House")[1])
    

def parse_REFIT(data_path, save_path):
    # read data

    data_path = data_path + "CLEAN_REFIT_081116/"

    # store house data in a dictionary keyed by house name and valued by a dictionary of appliances
    data = {}
    for file in os.listdir(data_path):
        if not file.endswith(".csv"):
            continue
        house_number = parse_name(file)
        # No device data for house 14
        if house_number == 14:
            continue
        df = process_dataframe(pd.read_csv(data_path + file), house_number)
        
        name = "REFIT_" + str(house_number)

        data[name] = df
        







   

    # save data
    save_to_pickle(data, save_path)
