import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import watts2kwh, save_to_pickle

def to_dict(df):
    return {"aggregate" : pd.DataFrame(df)}



def parse_HUE(data_path, save_path):
    residentials = pd.read_parquet(data_path).set_index("timestamp")
    # wH -> kWh
    residentials["energy"] = residentials["energy"] /1000
    residentials = residentials.copy()
    
    # save each house in a separate dataframe
    df_dict = {"house_"+str(id): to_dict(residentials["energy"]) for id, df_group in residentials.groupby('residential_id')}


    save_to_pickle(df_dict, save_path)
