import pandas as pd
from helper_functions import watts2kwh, save_to_pickle

######################DATASET INFO#########################################
# sampling rate: 1 hour
# unit: kWh
# households: 28
# no submeter data
# Location: Canada
# Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N3HGRN



def to_dict(df: pd.DataFrame) -> dict:
    return {"aggregate": pd.DataFrame(df)}


def parse_HUE(data_path: str, save_path: str) -> None:
    residentials = pd.read_parquet(data_path).set_index("timestamp")
    # Wh -> kWh
    residentials["energy"] = residentials["energy"] / 1000
    residentials = residentials.copy()

    # Create a dictionary with the data for each house
    data = {}
    for id in residentials["residential_id"].unique():
        data["HUE_" + str(id)] = {"aggregate" : residentials.loc[residentials["residential_id"] == id, "energy"]}

    # save each house in a separate dataframe
    df_dict = {f"HUE_{key}": to_dict(residentials["energy"]) for key, df_group in residentials.groupby("residential_id")}

    save_to_pickle(data, save_path)
