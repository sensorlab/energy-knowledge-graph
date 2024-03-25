from src.helper import *
import os
import pandas as pd
from pathlib import Path

######################DATASET INFO#########################################
# sampling rate: 1min
# length: 3.1 years
# unit: watts
# households: 50
# no submeter data
# Location: Portugal
# Source: https://osf.io/2ac8q/

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe
    """
    df = df.drop(columns=["Imin", "Imax", "Iavg", "Vmin", "Vmax", "Vavg", "Pmin", "Pmax", "Qmin", "Qmax", "Qavg", "PFmin", "PFmax", "PFavg", "miss_flag", "iid", "deploy"]).dropna()
    df = df.set_index("tmstp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("min").fillna(method="nearest", limit=5).dropna()  # if there is data within 5 minutes, fill it in else drop it

    return df


def parse_SUST1(data_path: str, save_path: str) -> None:
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"
    data_dict = {}
    for house in range(1, 51):
        name = "SUST1_" + str(house)
        tmp = {"aggregate": pd.DataFrame()}
        data_dict[name] = tmp
    data_path = data_path / "aggregate/"
    for folder in os.listdir(data_path):
        for file in os.listdir(data_path / folder):
            if file.endswith(".csv"):
                df = pd.read_csv(data_path / folder / file)
                # drop rows with missing data
                df = df[df["miss_flag"] == 0]
                # convert timestamp to datetime
                df["tmstp"] = pd.to_datetime(df["tmstp"])
                df.rename(columns={"Pavg": "aggregate"}, inplace=True)
                for iid in df["iid"].unique():
                    name = "SUST1_" + str(iid)
                    data_dict[name]["aggregate"] = pd.concat([data_dict[name]["aggregate"], preprocess_df(df[df["iid"] == iid])], axis=0)

    for house in data_dict:
        data_dict[house]["aggregate"] = data_dict[house]["aggregate"].sort_index()
        data_dict[house]["aggregate"] = data_dict[house]["aggregate"][~data_dict[house]["aggregate"].index.duplicated(keep="first")]

    save_to_pickle(data_dict, save_path)
