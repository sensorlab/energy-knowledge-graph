import pandas as pd

from helper_functions import save_to_pickle


######################DATASET INFO#########################################
# sampling rate: 1min
# unit: watts
# households: 1
# submetered: no
# Location: France
# Source: https://doi.org/10.24432/C58K54

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df.drop(columns=["country", "region", "lat", "lon", "tz", "voltage", "global_reactive_power", "global_intensity", "unmetered", "sub_metering_1", "sub_metering_2", "sub_metering_3"], inplace=True)
    df.rename(columns={"global_active_power": "aggregate"}, inplace=True)
    return df


def parse_UCIML(data_path: str, save_path: str) -> None:
    household = pd.read_parquet(data_path)
    df = preprocess_dataframe(household)
    houses = {}
    data = {}
    for col in df.columns:
        data[col] = pd.DataFrame(df[col])
    houses["UCIML_1"] = data

    save_to_pickle(houses, save_path)
