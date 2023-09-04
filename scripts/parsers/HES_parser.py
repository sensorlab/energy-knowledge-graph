import pandas as pd
import os
from helper_functions import watts2kwh, save_to_pickle


# read data from HES convert to kWh and save to dictionary
def parse_HES(data_path, save_path):
    device_dict = pd.read_pickle(data_path + "HES_processed.pkl")
        
    # Initialize an empty list to store device dataframes
    house_data = {}

    dfs = {}
    # Iterate over each device
    for device in device_dict:
        # Concatenate all daily dataframes for the current device
        # check if df is empty
        if len(device_dict[device]) == 0:
            continue
        if "basement bathroom" in device:
            continue
        device_df = pd.concat(device_dict[device], axis=0)
        # Reset the index (to handle any potential index related issues)
        device_df = device_df.reset_index()
        # Rename the columns to include the device name for uniqueness
        device_df.columns = ['date', device]
        
        
        # set index to date and sort
        device_df.set_index('date', inplace=True)
        device_df.sort_index(inplace=True)

        # Convert the power values from watts to kWh
        device_df = watts2kwh(device_df, 7/3600)

        # Add the current device dataframe to the dict of dataframes
        dfs[device] = device_df


    house_data['HES_1'] = dfs


    df_total = pd.Series(dtype='float64')

    # calculate total energy consumption
    for device in house_data['HES_1']:
        df_total = df_total.add(house_data['HES_1'][device][device], fill_value=0)

        
    # rename the column to 'aggregate'
    df_total = df_total.rename('aggregate')

    # add the aggregate to the house data
    house_data['HES_1']['aggregate'] = pd.DataFrame(df_total)

    save_to_pickle(house_data, save_path)    


if __name__ == "__main__":
    data_path = "../../data/HES/HES.pkl"
    save_path = "../../data/HES/HES_processed.pkl"
    parse_HES(data_path, save_path)