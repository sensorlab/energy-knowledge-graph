import pandas as pd
import os


from helper_functions import save_to_pickle, watts2kwh

# gets the number of the device from the filename
def getNumber(device : str) -> int:
    return int((device.split('.')[0]).split('_')[1])

# preproces the file by reading it, converting the timestamp to datetime and setting it as index
def preproces_file(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None, sep=" ")
    df[0] = pd.to_datetime(df[0], unit='s')
    df = df.set_index(0)
    df = watts2kwh(df, 6/3600)
    
    return df



# processes a house by reading all the files in the house folder and saving dfs for each device and returning them as a dictionary
def process_house(house_path : str) -> dict:
    data = {}
    # read the labels file to get the names of the columns
    lables = pd.read_csv(house_path+ "labels.dat", header=None, sep=" ")[1].values
    for device in os.listdir(house_path):
        if device.endswith(".dat"):
            if "channel" in device and "button" not in device:
                # get the number of the device from the filename
                number = getNumber(device)
                name = lables[number-1]
                df = preproces_file(house_path + device)
                data[name] = df

    return data


def parse_UKDALE(data_path, save_path):

    
    # dictionary of houses, each house is a dictionary of devices
    houses_data = {}
    for house in os.listdir(data_path):
        if "house" in house:
            name = "UKDALE_"+house.split("_")[1]
            houses_data[name] = process_house(data_path + "/" + house + "/")
        
    save_to_pickle(houses_data, save_path)

