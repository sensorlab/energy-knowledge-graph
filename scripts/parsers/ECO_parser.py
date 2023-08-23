import pandas as pd
import os
import pickle
from functools import reduce
from helper_functions import watts2kwh, save_to_pickle


# read data from ECO convert to kWh and save to dictionary
def get_house_data(file_path: str, device_mapping: dict):
    file_path_device = file_path + "/PLUGS"
    file_path_SM = file_path + "/SM"
    # dict to store appliance and aggregate consumption data
    house_data = {}
    # read device data
    for device in os.listdir(file_path_device):
        path = file_path_device +"/"+ device
        device_df = pd.DataFrame()
        # get device name from map
        device_name = device_mapping[int(device)]
        for f in os.listdir(path):
            if f.endswith(".csv"):
                df = pd.read_csv(path+"/"+f, header=None)
                if len(df) != 86400:
                    print(device_name, f, len(df))
                date_index = pd.date_range(start=f.split(".")[0], periods=len(df), freq="1S")
                df["date"] = date_index
                df = df.set_index("date")
                
                df[df==-1] = 0  
                df.rename(columns={0: device_name}, inplace=True)
                device_df = pd.concat([device_df, df], axis=0)
        
        
        device_df.sort_index(inplace=True)
        device_df = watts2kwh(device_df, 1/3600)

        house_data[device_name] = device_df

    # read total data
    total_df = pd.DataFrame()

    for f in os.listdir(file_path_SM):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(file_path_SM, f), header=None)
            df.drop(df.columns[1:], axis=1, inplace=True)
            df.columns = ["aggregate"]
            date_index = pd.date_range(start=f.split(".")[0], periods=len(df), freq='S')
            df.set_index(date_index, inplace=True)

            # ignore days with missing data
            if not (df == -1).any().any():
                total_df = pd.concat([total_df, df], axis=0)

    total_df = watts2kwh(total_df, 1/3600)
    house_data["aggregate"] = total_df


    return house_data




# return a dictionary of number to device name mapping for given house 
def get_device_map(house: str):

    # number to device name mapping
    device_map_house1 = {
        1: "Fridge",
        2: "Dryer",
        3: "Coffee machine",
        4: "Kettle",
        5: "Washing machine",
        6: "PC",
        7: "Freezer",
    }

    device_map_house2 = {
        1: "Tablet",
        2: "Dishwasher",
        3: "Air exhaust",
        4: "Fridge",
        5: "Entertainment",
        6: "Freezer",
        7: "Kettle",
        8: "Lamp",
        9: "Laptops",
        10: "Stove",
        11: "TV",
        12: "Stereo",
    }

    device_map_house3 = {

        1: "Tablet",
        2: "Freezer",
        3: "Coffee machine",
        4: "PC",
        5: "Fridge",
        6: "Kettle",
        7: "Entertainment",
    }

    device_map_house4 = {
        1: "Fridge",
        2: "Kitchen appliances",
        3: "Lamp",
        4: "Stereo",
        5: "Freezer",
        6: "Tablet",
        7: "Entertainment",
        8: "Microwave",
    }

    device_map_house5 = {
        1: "Tablet",
        2: "Coffee machine",
        3: "Fountain",
        4: "Microwave",
        5: "Fridge",
        6: "Entertainment",
        7: "PC",
        8: "Kettle",
    }

    device_map_house6 = {
        1: "Lamp",
        2: "Laptop",
        3: "Router",
        4: "Coffee machine",
        5: "Entertainment",
        6: "Fridge",
        7: "Kettle",
    }

    if house == "HOUSE1":
        return device_map_house1
    elif house == "HOUSE2":
        return device_map_house2
    elif house == "HOUSE3":
        return device_map_house3
    elif house == "HOUSE4":
        return device_map_house4
    elif house == "HOUSE5":
        return device_map_house5
    elif house == "HOUSE6":
        return device_map_house6
    else:
        print("Invalid house name")
        return None
        
# 
def parse_ECO(file_path: str, path_to_save: str):

        
    houses_data = {}

    for house in os.listdir(file_path):
        mapping = get_device_map(house)
        name = "ECO_"+house[-1]
        houses_data[name] = get_house_data(file_path+"/"+house, mapping)


    save_to_pickle(houses_data, path_to_save)



if __name__ == "__main__":
    parse_ECO("data/ECO", "data/ECO_parsed.pkl")



