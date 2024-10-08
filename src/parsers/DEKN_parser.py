import pandas as pd
from pathlib import Path

from src.helper import save_to_pickle
######################DATASET INFO#########################################
# sampling rate: 1min
# length: 2.5 years
# unit: kWh
# households: 5
# submetered
# Location: Germany
# Source: https://data.open-power-system-data.org/household_data/2020-04-15
# https://data.open-power-system-data.org/household_data/


def parse_DEKN(data_path: str, save_path: str) -> None:
    """
    Parse the DEKN dataset and save to a pickle file
    ## Parameters
    data_path : The path to the DEKN dataset
    save_path : The path to save the parsed data

    """
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"
    df = pd.read_csv(data_path / "household_data_1min_singleindex_filtered.csv", low_memory=False)

    # drop unnecessary columns
    df = df.drop(columns=["utc_timestamp", "interpolated"])

    # convert timestamp to datetime and set as index
    df["cet_cest_timestamp"] = df["cet_cest_timestamp"].apply(lambda x: x.split("+")[0])
    df["cet_cest_timestamp"] = pd.to_datetime(df["cet_cest_timestamp"], format="%Y-%m-%dT%H:%M:%S")
    df = df.set_index("cet_cest_timestamp")
    df.sort_index(inplace=True)

    # handle duplicates from daylight savings time change
    df = df[~df.index.duplicated(keep="first")]

    # Extract household identifiers
    households = set(column.split("_")[2] for column in df.columns)

    # Create a dictionary of dataframes, one for each household
    data_dict = {}

    for household in households:
        # Filter columns relevant to this household
        relevant_columns = [col for col in df.columns if household in col]
        temp_df = df[relevant_columns].copy()

        # Rename columns to remove the prefix and retain the device name
        rename_dict = {col: col.replace(f"DE_KN_{household}_", "") for col in relevant_columns}
        temp_df.rename(columns=rename_dict, inplace=True)
        temp_df.rename(columns={"cet_cest_timestamp": "timestamp", "grid_import": "aggregate"}, inplace=True)
        if "grid_export" in temp_df.columns:
            temp_df.drop(columns=["grid_export"], inplace=True)
        if "pv" in temp_df.columns:
            temp_df.drop(columns=["pv"], inplace=True)

        data = {}
        # get name of household
        name = "DEKN_" + str(household[-1])
        # create dataframe for each appliance
        for c in temp_df.columns:
            curr_df = temp_df[c].copy()
            # data is cumulative, so get the difference
            curr_df = curr_df.diff(periods=1)
            curr_df = curr_df.dropna()

            data[c] = pd.DataFrame(curr_df)

        data_dict[name] = data

    save_to_pickle(data_dict, save_path)
