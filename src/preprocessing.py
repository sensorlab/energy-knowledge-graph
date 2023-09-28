import pandas as pd
import os
import re
from tqdm import tqdm
import pickle
from collections import Counter
import concurrent.futures
import argparse
from helper import preprocess_string

    
def process_dictionary(data: dict, values=0) -> pd.DataFrame:

    ignored_devices = [
        "light",
        "outlet",
        "sockets",
        "lamp",
        "plug",
        'CE appliance'
        'kettle/toaster',
        'dehumidifier/heater',
        'HairDryer-Straightener',
        'Office Desk',
        'heat basement',
        'set top box',
        'subpanel',
    ]
    dfs = []

    for device in data:
        # ignore devices
        if any(ignored_device in device.lower() for ignored_device in ignored_devices):
            continue

        # preprocess device name
        device_name = preprocess_string(device)
        
        df = data[device]

        # rename column to standardized device name
        df.columns = [device_name]
        if df.max().max() < 2:
            print("device with zeros: ", device_name)
            continue

        time_diffs = df.index.to_series().diff()
        median_interval = time_diffs.median()

        # if there is less than 3 days of data drop the device
        if len(df) < (3*24 * 60 * 60) / median_interval.total_seconds():
            print("less than 3 days of data for device: ", device_name)
            continue
        dfs.append(df)

    # concatenate all dataframes
    df = pd.concat(dfs, axis=1)

    # resample to 8s
    df = df.resample("8s").fillna(method="nearest", limit=4)

    # check for gaps in data TODO do this in 

    # drop rows with NaN values
    df.dropna(inplace=True)
    
    # handle negative values
    df[df<0] = 0

    # treshold in watts
    treshold = 5
    if values == 0:
        # put 1 if device is on and 0 if device is off
        for c in df.columns:
            if c == "aggregate":
                continue
            # if power is less than treshold device is off
            df[c] = (df[c] > treshold).astype(int)

    # find duplicate columns
    column_counts = Counter(df.columns)
    duplicates = [col for col, count in column_counts.items() if count > 1]
    # Sum duplicate columns
    for duplicate in duplicates:
        duplicate_cols = [col for i, col in enumerate(df.columns) if col == duplicate]
        df[duplicate] = df[duplicate_cols].sum(axis=1)
        # Drop other duplicate columns if needed
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    
    return df
    


def process_dataset(dataset_path):
    data = pd.read_pickle(dataset_path)
    # print(dataset_path)
    for house in data:
        data[house] = process_dictionary(data[house])
        
    return data


def get_data(path : str, labels_path : str, values=0):
        
    # path = "./Energy_graph/data/processed_watts/"
    dataset_paths = [os.path.join(path, dataset) for dataset in os.listdir(path) if dataset.endswith('.pkl')]
        
    cpu_count = int(os.cpu_count() / 2)
    data_dict = {}

    with tqdm(total=len(dataset_paths), desc="Processing datasets", unit="dataset") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {executor.submit(process_dataset, dataset_path): dataset_path for dataset_path in dataset_paths}
            
            for future in concurrent.futures.as_completed(futures):
                dataset_path = futures[future]
                try:
                    processed_data = future.result()
                    data_dict.update(processed_data)
                except Exception as e:
                    print(f"Dataset {dataset_path} generated an exception: {e}")
                
                progress_bar.update(1)

    labels = pd.read_pickle(labels_path)
    labels.sort()

    return data_dict

def create_windows(data : dict, labels_path: str, save_path : str, time_window=2700, upper_bound=pd.Timedelta(seconds=32), max_gap = pd.Timedelta(seconds=3600)):
    """Creates windows of time_window seconds from the data and discards windows with gaps of more than 1h or 15 gaps of 32 seconds or more"""
    labels = pd.read_pickle(labels_path)
    # windows = []
    X_Y = [] # list of tuples (X, Y)
    skip_count_1 = 0
    skip_count_2 = 0
    total_count = 0

    for df in tqdm(data.values()):
        for i in range(0, len(df) - time_window, time_window + 1):
            window = df.iloc[i:i + time_window]
            total_count += 1
            # if there is a gap of more than max_gap skip the window
            time_diffs = window.index.to_series().diff().dropna()
            if  (time_diffs >= max_gap).any():
                skip_count_1 += 1
                continue
            # if there are more than 15 gaps of upper_bound or more skip the window
            if len(time_diffs[time_diffs > upper_bound]) > 15:
                skip_count_2 += 1
                continue

            x = window["aggregate"].values
            devices = [False] * len(labels)
            # check if device is on in the window
            for c in window.columns:
                if c == "aggregate":
                    continue
                on = (window[c] > 0)
                ix = labels.index(c)
                devices[ix] = on.any()

            X_Y.append((x, devices))
            


            # windows.append(window)

    print("Total windows: ", total_count, "Skipped windows due to 30min gap: ", skip_count_1, "Skipped windows due to 15 gaps of 32s or more: ", skip_count_2 ,"Procentage skipped: ", (skip_count_1+skip_count_2) / total_count * 100)
    with open(save_path+ f"/X_Y_wsize{time_window}_upper{int(upper_bound.total_seconds())}_gap{int(max_gap.total_seconds())}.pkl", "wb") as f:
        pickle.dump(X_Y, f, protocol=pickle.HIGHEST_PROTOCOL)


# example command line call:
# python preprocessing.py --path_to_base "/your/path" --time_window 3000 --upper_bound 40 --max_gap 4000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process energy graph data.")
    
    # Argument for the path to the base folder
    parser.add_argument("--path_to_base", type=str, default=".", help="Path to base folder")
    
    # Argument for time window
    parser.add_argument("--time_window", type=int, default=2700, help="Time window in seconds")
    
    # Argument for upper bound
    parser.add_argument("--upper_bound", type=int, default=32, help="Upper bound in seconds")
    
    # Argument for max gap
    parser.add_argument("--max_gap", type=int, default=3600, help="Max gap in seconds")
    
    args = parser.parse_args()

    # Initialize paths
    path_to_base = args.path_to_base
    path = path_to_base + "/data/processed_watts/"
    labels_path = path_to_base + "/data/labels_new.pkl"
    save_path = path_to_base + "/data/training_data/processed/"
    
    # Initialize parameters
    time_window = args.time_window
    upper_bound = pd.Timedelta(seconds=args.upper_bound)
    max_gap = pd.Timedelta(seconds=args.max_gap)
    
    # Print parameters
    print("Path to base folder: ", path_to_base)
    print("Path to save windows: ", save_path)
    print("Time window: ", time_window, "rows | ", time_window*8, "seconds")
    print("Upper bound: ", upper_bound)
    print("Max gap: ", max_gap)

    # Get data and create windows
    data = get_data(path, labels_path)
    create_windows(data, labels_path, save_path, time_window, upper_bound, max_gap)