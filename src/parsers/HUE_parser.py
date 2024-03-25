import pandas as pd
from pathlib import Path
from src.helper import save_to_pickle

######################DATASET INFO#########################################
# sampling rate: 1 hour
# length: 3 years
# unit: kWh
# households: 28
# no submeter data
# Location: Canada
# Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N3HGRN



def to_dict(df: pd.DataFrame) -> dict:
    return {"aggregate": pd.DataFrame(df)}


def parse_HUE(data_path: str, save_path: str) -> None:
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    data = pd.read_parquet(data_path).set_index("timestamp")
    # Wh -> kWh
    data["energy"] = data["energy"] / 1000
    data = data.copy()

    # Create a dictionary with the data for each house
    data_dict = {}
    for id in data["residential_id"].unique():
        data_dict["HUE_" + str(id)] = {"aggregate" : data.loc[data["residential_id"] == id, "energy"]}

    save_to_pickle(data_dict, save_path)
