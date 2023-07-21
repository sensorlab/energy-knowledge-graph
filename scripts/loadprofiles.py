import os
from tqdm import tqdm
import pandas as pd
import pickle
import sys



def calculate_loadprofiles(df):
    # resample to daily and hourly
    hourly = df.resample('H').sum()
    daily = df.resample('D').sum()


        
    # daily load profile
    loadprofile_daily = hourly.groupby(hourly.index.hour).mean()

    # weekly load profile
    loadprofile_weekly = daily.groupby(daily.index.dayofweek).mean()

    # monthly load profile
    loadprofile_monthly = daily.groupby(daily.index.day).mean()

    # save to dictioanry
    loadprofiles = {
        "daily": loadprofile_daily.values,
        "weekly": loadprofile_weekly.values,
        "monthly": loadprofile_monthly.values
    }
    return loadprofiles



if len(sys.argv) < 3:
    print("Usage: python loadprofiles.py <path to data> <path to save folder>")
    sys.exit(1)
elif len(sys.argv) == 3:
    print("Processing data from " + sys.argv[1] + " and saving to " + sys.argv[2])
    data_path = sys.argv[1]
    save_folder = sys.argv[2]



for dataset in tqdm(os.listdir(data_path)):
    name = dataset.split(".")[0]
    data_dict = pd.read_pickle(data_path + dataset)
    loadprofiles = {}
    for house in data_dict:
        house_lp = {}
        for device in data_dict[house]:
            house_lp[device] = calculate_loadprofiles(data_dict[house][device])

        loadprofiles[house] = house_lp
    with open(save_folder + name + "_loadprofiles.pkl", 'wb') as f:
        pickle.dump(loadprofiles, f, pickle.HIGHEST_PROTOCOL)
    

