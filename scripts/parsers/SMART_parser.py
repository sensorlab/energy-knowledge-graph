import pandas as pd
import os

from helper_functions import save_to_pickle

# convert to kWh
def convert2kWh(df):
    df['power[kW]'] = df['power[kW]'] * 0.25 # 15 min interval -> *4
    df.columns = ['time', 'power[kWh]']
    return df

# read the file and convert to kWh 
def process_file(file_path):
    df = pd.read_csv(file_path, header=None, names=["time", "power[kW]"])
    df = convert2kWh(df)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index('time')
    return df

# get apartment name from file path
def apartment_name(file_path : str):
    return file_path.split("_")[0]


def sort_key(s):
    return int(s[3:])  # Extract the number part of the string and convert to int

def get_apartment_names(folder_path):
    # sets to store apartment names for each year
    apartments = []


    for file in os.listdir(folder_path + "2015"):
        if not file.endswith(".csv"):
            continue
        # read apartment name from file name
        apt_name = apartment_name(file)
        apartments.append(apt_name)
        
            

    # sort apartments by number
    apartments = sorted(list(apartments), key=sort_key)
    return apartments

def parse_SMART(data_path, save_path):
        
    apartments = get_apartment_names(data_path)

    # stores data for all apartments with the following structure: {apartment_name: energy_data}
    data_for_all_apartments = {}
    # these 6 apartments are missing in 2014 data but appear in 2015 and 2016
    apartments_missing2014 = ['Apt65', 'Apt6', 'Apt21', 'Apt112', 'Apt94', 'Apt3']
   
    for apt in apartments:
        # handle missing 2014 data
        if apt in apartments_missing2014:
            df_2015 = process_file(data_path + "2015/" + apt + "_2015.csv")
            df_2016 = process_file(data_path + "2016/" + apt + "_2016.csv")
            df = pd.concat([df_2015, df_2016])
        else:
            df_2014 = process_file(data_path + "2014/" + apt + "_2014.csv")
            df_2015 = process_file(data_path + "2015/" + apt + "_2015.csv")
            df_2016 = process_file(data_path + "2016/" + apt + "_2016.csv")
            df = pd.concat([df_2014, df_2015, df_2016])
            
        

        data_for_all_apartments[apt] = df


    save_to_pickle(data_for_all_apartments, save_path)
        



