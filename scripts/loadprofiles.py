import os
from tqdm import tqdm
import pandas as pd
import pickle
import sys
import concurrent.futures
import multiprocessing

# watts to kWh given data frequency as a fraction of an hour (e.g. 0.5 for half-hourly data)
def watts2kwh(df, data_frequency):
    df = df/1000 * data_frequency
    return df


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

def process_dataset(dataset):
    data_dict = pd.read_pickle(os.path.join(data_path, dataset))
    loadprofiles = {}
    for house in data_dict:
        house_lp = {}
        for device in data_dict[house]:
            # these datasets are already in kWh
            if "ECDUY" in  house or "DEKN" in house or "HUE" in house:
                house_lp[device] = calculate_loadprofiles(data_dict[house][device])
            else:
                # get sampling rate for kWh conversion
                time_deltas = data_dict[house][device].index.to_series().diff().dropna()
                median_time_delta = time_deltas.median()
                sampling_rate = median_time_delta.total_seconds()/3600
                house_lp[device] = calculate_loadprofiles(watts2kwh(data_dict[house][device], sampling_rate))
            
        loadprofiles[house] = house_lp

    with open(os.path.join(save_folder, dataset.split(".")[0] + "_loadprofiles.pkl"), 'wb') as f:
        pickle.dump(loadprofiles, f, pickle.HIGHEST_PROTOCOL)

    return dataset.split(".")[0], loadprofiles

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python loadprofiles.py <path to data> <path to save folder>")
        sys.exit(1)
    elif len(sys.argv) == 3:
        print("Processing data from " + sys.argv[1] + " and saving to " + sys.argv[2])
        data_path = sys.argv[1]
        save_folder = sys.argv[2]

    dataset_paths = [dataset for dataset in os.listdir(data_path) if dataset.endswith('.pkl')]
    queue = multiprocessing.Manager().Queue()



    cpu_count = int(os.cpu_count()/2)
    data_dict = {}
    with tqdm(total=len(dataset_paths), desc="Processing datasets", unit="dataset") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(process_dataset, dataset) for dataset in dataset_paths]

            for future in concurrent.futures.as_completed(futures):
                dataset_name, dataset_loadprofile = future.result()
                data_dict.update(dataset_loadprofile)

                progress_bar.update(1)  # update progress bar


    with open(save_folder + "/" +"merged_loadprofiles.pkl", 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
        


