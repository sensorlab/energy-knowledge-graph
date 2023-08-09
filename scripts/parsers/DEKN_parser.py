import pandas as pd
from helper_functions import save_to_pickle


# https://data.open-power-system-data.org/household_data/
def parse_DEKN(data_path : str, save_path : str):
    df = pd.read_csv(data_path+"household_data_15min_singleindex_filtered.csv")

    # drop unnecessary columns
    df =df.drop(columns=["utc_timestamp", "interpolated"])

    # convert timestamp to datetime and set as index 
    df["cet_cest_timestamp"] = df["cet_cest_timestamp"].apply(lambda x: x.split("+")[0])
    df["cet_cest_timestamp"] = pd.to_datetime(df["cet_cest_timestamp"], format="%Y-%m-%dT%H:%M:%S")
    df = df.set_index("cet_cest_timestamp")

    # handle duplicates from daylight savings time change
    df = df[~df.index.duplicated(keep='first')]


    
    # Extract household identifiers
    households = set(column.split('_')[2] for column in df.columns)

    # Create a dictionary of dataframes, one for each household
    dfs = {}

    for household in households:
        # Filter columns relevant to this household
        relevant_columns = [col for col in df.columns if household in col]
        temp_df = df[relevant_columns].copy()

        # Rename columns to remove the prefix and retain the device name
        rename_dict = {col: col.replace(f"DE_KN_{household}_", "") for col in relevant_columns}
        temp_df.rename(columns=rename_dict, inplace=True)
        temp_df.rename(columns={'cet_cest_timestamp': 'timestamp', "grid_import": "aggregate"}, inplace=True)
        if "grid_export" in temp_df.columns:
            temp_df.drop(columns=['grid_export'], inplace=True)
        if "pv" in temp_df.columns:
            temp_df.drop(columns=['pv'], inplace=True)

        data = {}
        # get name of household
        name ="DEKN_" +str(household[-1])
        # create dataframe for each appliance
        for c in temp_df.columns:
            data[c] = pd.DataFrame(temp_df[c].dropna())
            
        dfs[name] = data


        save_to_pickle(dfs, save_path)