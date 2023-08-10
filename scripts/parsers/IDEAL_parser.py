
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from helper_functions import *

def watts2kwh(df, data_frequency):
    df = df/1000 * data_frequency
    return df
def read_and_preprocess_df(path):
    df = pd.read_csv(path, header=None, names=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # set timestamp as index
    df = df.set_index("timestamp")
    df.sort_index(inplace=True)
    # resample to 7s and forward fill up to 35s
    df = df.resample("7s").ffill(limit=7).dropna()

    # convert to kWh
    df = watts2kwh(df, 7/3600)
    return df

# get house name and appliance name from file name
def parse_name(file_name : str):
    file_name = file_name.split("_")
    house_name = file_name[0].replace("home", "IDEAL_")
    appliance_name = file_name[3]
    if appliance_name == "electric-mains":
        appliance_name = "aggregate"

    if appliance_name == "electric-appliance":
        appliance_name = file_name[4].split(".")[0]
    return house_name, appliance_name
# process a single house
def process_house(house, file_list, data_path):
    house_data = {}
    for file in file_list:
        _, label, df = process_file(file, data_path)
        house_data[label] = df
    return house, house_data
# process a single file for a house
def process_file(file,data_path):
    house, label = parse_name(file)
    return house, label, read_and_preprocess_df(data_path + "data_merged/" + file)


def unpack_and_process(p):
    return process_house(*p)

def parse_IDEAL(data_path: str, save_path : str):

    data_dict = {}
    files_grouped_by_home = defaultdict(list)
    # get files for electricity consumption
    files = [file for file in os.listdir(data_path + "data_merged/") if ("electric-appliance" in file or "electric-mains" in file) and "home223" not in file]
    # group files by house
    for file in files:
        house, _ = parse_name(file)
        files_grouped_by_home[house].append(file)

    total_houses = len(files_grouped_by_home)

    print("Processing houses...")
    # here we use half of the cpu cores to process the data you can change this if you want
    cpu_count = int(os.cpu_count()/2)
    # process the houses in parallel
    with ProcessPoolExecutor(max_workers=cpu_count) as executor, tqdm(total=total_houses, desc="Processing houses", unit="house") as t:
        args = ((house, files_grouped_by_home[house], data_path) for house in files_grouped_by_home)
        
        for house_name, house_data in executor.map(unpack_and_process, args):
            data_dict[house_name] = house_data
            t.update(1)

    # save the data to a dictonary
    save_to_pickle(data_dict, save_path)

