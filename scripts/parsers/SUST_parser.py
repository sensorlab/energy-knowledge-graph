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
    # localize to Lisbon timezone and handle DST
    df_aggregate.index =  df_aggregate.index.tz_localize("UTC", ambiguous="NaT").tz_convert("Europe/Lisbon")
    df_aggregate = df_aggregate[df_aggregate.index.notna()]


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
            # convert to Lisbon timezone and handle DST
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC", ambiguous="NaT").tz_convert("Europe/Lisbon")
                df = df[df.index.notna()]
            else:   
                df.index =  df.index.tz_convert("Europe/Lisbon")
                # Handle duplicate indices due to DST: keep the first occurrence, drop the others
                df = df[~df.index.duplicated(keep='first')]



            data_dict[parse_name(file)] = df

    data = {
        "SUST_1": data_dict
    }
    # save to pickle
    save_to_pickle(data, save_path)

