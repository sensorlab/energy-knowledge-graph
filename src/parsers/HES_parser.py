import pandas as pd
from src.helper import  save_to_pickle
from pathlib import Path

######################DATASET INFO#########################################
# sampling rate: 7s
# length: 5 months
# unit: watts
# households: 1
# submetered: yes
# Location: Canada
# Source: https://github.com/ETSSmartRes/HES-Dataset


# read data from HES convert to uniform format and save to dictionary
def parse_HES(data_path: str, save_path: str) -> None:
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    device_dict = pd.read_pickle(data_path / "HES_processed.pkl")

    data_dict = {}

    data = {}
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
        device_df.columns = ["date", device]
        device_df["date"] = device_df["date"].dt.to_timestamp()
        device_df["date"] = pd.to_datetime(device_df["date"])
        # set index to date and sort
        device_df.set_index("date", inplace=True)
        device_df.sort_index(inplace=True)

        # remove duplicates
        device_df = device_df[~device_df.index.duplicated(keep="first")]

        # Add the current device dataframe to the dict of dataframes
        data[device] = device_df

    data_dict["HES_1"] = data

    df_total = pd.Series(dtype="float64")

    # calculate total energy consumption
    for device in data_dict["HES_1"]:
        df_total = df_total.add(data_dict["HES_1"][device][device], fill_value=0)

    # rename the column to 'aggregate'
    df_total = df_total.rename("aggregate")

    # add the aggregate to the house data
    data_dict["HES_1"]["aggregate"] = pd.DataFrame(df_total)

    save_to_pickle(data_dict, save_path)

