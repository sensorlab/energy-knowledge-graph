import pandas as pd
from helper_functions import watts2kwh, save_to_pickle


def to_dict(df: pd.DataFrame) -> dict:
    return {"aggregate": pd.DataFrame(df)}


def parse_HUE(data_path: str, save_path: str) -> None:
    residentials = pd.read_parquet(data_path).set_index("timestamp")
    # Wh -> kWh
    residentials["energy"] = residentials["energy"] / 1000
    residentials = residentials.copy()

    # save each house in a separate dataframe
    df_dict = {f"HUE_{key}": to_dict(residentials["energy"]) for key, df_group in residentials.groupby("residential_id")}

    save_to_pickle(df_dict, save_path)
