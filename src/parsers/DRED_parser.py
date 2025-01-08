import pandas as pd
from src.helper import save_to_pickle
from pathlib import Path

######################DATASET INFO#########################################
# sampling rate: 1s
# length: 6 months
# unit: watts 
# households: 1
# submetered
# Location: Netherlands
# Source: https://www.st.ewi.tudelft.nl/~akshay/dred/


def parse_DRED(data_path: str, save_path: str) -> None:
    """
    Parse the DRED dataset and save to a pickle file
    ## Parameters
    data_path : The path to the DRED dataset
    save_path : The path to save the parsed data
    """
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"
    df = pd.read_csv(data_path / "All_data.csv", skiprows=1).drop(columns=["unknown"])

    # rename time column and mains to aggregate
    df = df.rename(columns={"Unnamed: 0": "time", "mains": "aggregate"})

    # drop missing data
    df = df.dropna()

    # fix time so that it can be converted to date time
    df["time"] = df["time"].astype(str)
    df["time"] = df["time"].str.split("+").str[0]

    df["time"] = pd.to_datetime(df["time"])

    # set time as index
    df.reset_index(drop=True, inplace=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    #check for duplicates
    df = df[~df.index.duplicated(keep="first")]

    # split into appliances
    data = {}
    for c in df.columns:
        data[c] = pd.DataFrame(df[c])

    data_dict = {"DRED_1": data}

    save_to_pickle(data_dict, save_path)
