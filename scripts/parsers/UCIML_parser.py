import pandas as pd

from helper_functions import save_to_pickle


def preprocess_dataframe(df):
    df = df.copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df.drop(columns=["country", "region", "lat", "lon", "tz", "voltage", "global_reactive_power", "global_intensity", "unmetered"], inplace=True)
    return df


def parse_UCIML(data_path, save_path):

    household = pd.read_parquet(data_path)
    df = preprocess_dataframe(household)
    houses = {}
    data = {}
    for col in df.columns:
        data[col] = pd.DataFrame(df[col])
    houses["UCIML_1"] = data

    save_to_pickle(houses, save_path)
