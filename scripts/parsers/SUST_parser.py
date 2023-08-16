import pandas as pd
import os
from helper_functions import save_to_pickle



def parse_name(file_name: str):
    """
    Parse the file name to get the appliance name
    """
    # appliance name
    appliance_name = file_name.split(".")[0].split("_")[1]
    # date
    return appliance_name

def parse_SUST(data_path : str, save_path : str):
    # aggregate consumption data
    df_aggregate = pd.DataFrame()
    for file in os.listdir(data_path + "aggregate"):
        if file.endswith(".csv"):
            df_aggregate = pd.concat([df_aggregate,(pd.read_csv(data_path+"aggregate/" + file))])

    # set timestamp as idnex
    df_aggregate["timestamp"] = pd.to_datetime(df_aggregate["timestamp"])
    df_aggregate.set_index("timestamp", inplace=True)

    # drop unnecessary columns
    df_aggregate.drop(columns=['Unnamed: 0', "Q","V","I"], inplace=True)

    df_aggregate.rename(columns={"P":"power"}, inplace=True)
    # save to dictonary
    data_dict = {"aggregate":df_aggregate}


    
    # appliance consumption data
    for file in os.listdir(data_path+"appliances/"):
        if file.endswith(".csv"):
            df = pd.read_csv(data_path + "appliances/" + file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            data_dict[parse_name(file)] = df

    data = {
        "SUST_1": data_dict
    }
    # save to pickle
    save_to_pickle(data, save_path)

