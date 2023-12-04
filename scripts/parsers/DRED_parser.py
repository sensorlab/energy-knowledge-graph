import pandas as pd
from helper_functions import save_to_pickle

def parse_DRED(data_path : str, save_path : str):
    df = pd.read_csv(data_path+"All_data.csv", skiprows=1).drop(columns=["unknown"])

    # rename time column and mains to aggregate
    df = df.rename(columns={'Unnamed: 0':'time','mains':'aggregate'})

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


    # split into appliances
    data = {}
    for c in df.columns:
        data[c] = pd.DataFrame(df[c])


    res = {
        "DRED_1": data
    }
    
    save_to_pickle(res, save_path)