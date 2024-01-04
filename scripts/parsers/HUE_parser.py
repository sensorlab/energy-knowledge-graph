import pandas as pd
from helper_functions import watts2kwh, save_to_pickle

######################DATASET INFO#########################################
# sampling rate: 1 hour
# unit: kWh
# households: 28
# no submeter data
# Location: Canada
# Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N3HGRN


def to_dict(df):
    return {"aggregate" : pd.DataFrame(df)}



def parse_HUE(data_path, save_path):
    residentials = pd.read_parquet(data_path).set_index("timestamp")
    # wH -> kWh
    residentials["energy"] = residentials["energy"] /1000
    residentials = residentials.copy()
        
    # Create a dictionary with the data for each house
    data = {}
    for id in residentials["residential_id"].unique():
        data["HUE_" + str(id)] = {"aggregate" : residentials.loc[residentials["residential_id"] == id, "energy"]}


    save_to_pickle(data, save_path)
