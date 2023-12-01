import pandas as pd
import os
from tqdm import tqdm
import pickle
import numpy as np
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
from helper import preprocess_string


RANDOM_SEED = 42


def sample_normal_within_range(mu=7, sigma=5, a=1, b=35, n=1):
    samples = []
    while len(samples) < n:
        value = round(np.random.normal(mu, sigma))
        if a <= value <= b:
            samples.append(value)
    return np.array(samples)[0]

def process_dataset(dataset, path, time_window, upper_bound, max_gap):
    devices_processed_local = {}
    if not dataset.endswith(".pkl"):
        return devices_processed_local
    data = pd.read_pickle(path + dataset)
    for house in data:
        for device in data[house]:
            if device == "aggregate":
                continue
            name = preprocess_string(device)
            if name not in devices_processed_local:
                devices_processed_local[name] = []
            curr_device = process_data(data[house][device], time_window, upper_bound, max_gap)
            devices_processed_local[name].extend(curr_device)
    return devices_processed_local

def process_data(df : pd.DataFrame, time_window, upper_bound, max_gap) -> list:    

    df = df.resample("8S").fillna(method="nearest", limit=4)
    df.fillna(0, inplace=True)
    # handle negatve values
    df[df<0] = 0 # TODO FIX
   
   
    windows = []
    for i in range(0, len(df) - time_window, time_window + 1):
        window = df.iloc[i:i + time_window]
        

        # if there is a gap of more than max_gap skip the window
        time_diffs = window.index.to_series().diff().dropna()
        if  (time_diffs >= max_gap).any():
            continue
        # if there are more than 15 gaps of upper_bound or more skip the window
        if len(time_diffs[time_diffs > upper_bound]) > 15:
            continue 
        # skip if the device is always off
        if window.max().max() < 5:
            # print("skipping window with zeros: ", window.max().max())
            continue
        window.reset_index(drop=True, inplace=True)
        
        windows.append(window)
    
    return windows


def get_device_windows(path, time_window, upper_bound, max_gap):
    devices_processed = {}
    datasets = os.listdir(path)
    
    with ProcessPoolExecutor(max_workers=4) as executor:  # Change max_workers as needed
        futures = {executor.submit(process_dataset, dataset, path, time_window, upper_bound, max_gap): dataset for dataset in datasets}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            dataset = futures[future]
            try:
                result = future.result()
                # Aggregate the results
                for name, windows in result.items():
                    if name not in devices_processed:
                        devices_processed[name] = []
                    devices_processed[name].extend(windows)
            except Exception as e:
                print(f"Failed to process dataset {dataset}. Error: {e}")

    return devices_processed


def process_dataset(dataset, path, time_window, upper_bound, max_gap):
    devices_processed_local = {}
    if not dataset.endswith(".pkl"):
        return devices_processed_local
    data = pd.read_pickle(path + dataset)
    for house in data:
        for device in data[house]:
            if device == "aggregate":
                continue
            name = preprocess_string(device)
            if name not in devices_processed_local:
                devices_processed_local[name] = []
            curr_device = process_data(data[house][device], time_window, upper_bound, max_gap)
            devices_processed_local[name].extend(curr_device)
    return devices_processed_local



# sum(devices) = aggregate
def generate_syn_ideal(devices_processed, num_windows, device_list):
    windows = []
    for i in tqdm(range(num_windows)):
        # get number of devices sampled from normal distribution
        nm_device = sample_normal_within_range()
        # randomly select devices from the list of devices
        selected_devices = (random.sample(device_list, nm_device))
    
        df = pd.DataFrame()
        # choose random windows from the selected devices
        for device in selected_devices:
            curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
            
            # if the device is never on choose another window
            while curr_df.max().max() == 0:
                curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
            curr_df.columns = [device]
            df = pd.concat([df, curr_df], axis=1)
        df["aggregate"] = df.sum(axis=1)
        count = 0
        # if the aggregate consumption is less than 20w try to choose other windows for devices if after 20 tries aggregate still under 20w choose other devices
        while df["aggregate"].median() < 20 or df["aggregate"].mean() < 20 or df["aggregate"].max() > 50000:
            count += 1
            if count > 20:
                nm_device = sample_normal_within_range()
                selected_devices = (random.sample(device_list, nm_device))
                count = 0
            df = pd.DataFrame()
            for device in selected_devices:
                curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
                
                
                while curr_df.max().max() == 0:
                    curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
                curr_df.columns = [device]
                df = pd.concat([df, curr_df], axis=1)
            df["aggregate"] = df.sum(axis=1)
    
        treshold = 5
        for c in df.columns:
            if c == "aggregate":
                continue

            df[c] = (df[c] > treshold).astype(int)
        windows.append(df)
    return windows
# sum(devices) < aggregate
def generate_syn_unmetered(devices_processed, num_windows, device_list):
    windows = []
    for i in tqdm(range(num_windows)):
        # get number of devices sampled from normal distribution
        nm_device = sample_normal_within_range()

        # randomly select devices from the list of devices
        selected_devices = (random.sample(device_list, nm_device))
        # choose random windows from the selected devices
        df = pd.DataFrame()
        for device in selected_devices:
            curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
            
            # if the device is never on choose another window
            while curr_df.max().max() == 0:
                curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
            curr_df.columns = [device]
            df = pd.concat([df, curr_df], axis=1)
        # calculate the aggregate consumption
        df["aggregate"] = df.sum(axis=1)
        count = 0
        # if the aggregate consumption is less than try to choose other windows for devices else choose other devices
        while df["aggregate"].median() < 20 or df["aggregate"].mean() < 20:
            count += 1
            if count > 20:
                nm_device = sample_normal_within_range()
                selected_devices = (random.sample(device_list, nm_device))
                count = 0
            df = pd.DataFrame()

            for device in selected_devices:
                curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
                
                
                while curr_df.max().max() == 0:
                    curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
                curr_df.columns = [device]
                df = pd.concat([df, curr_df], axis=1)
            df["aggregate"] = df.sum(axis=1)
    
        # pick random 1-10 devices and add them to the aggregate consumption
        nm_device = sample_normal_within_range(mu=3, sigma=2, a=1, b=10)
        selected_devices = (random.sample(device_list, nm_device))
        unmetered = pd.DataFrame(0, index=range(len(curr_df)), columns=['aggregate'])
        for device in selected_devices:
            curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
            while curr_df.max().max() == 0:
                curr_df = devices_processed[device][random.randint(0, len(devices_processed[device]) - 1)]
            curr_df.columns = [device]
            unmetered['aggregate'] += curr_df[device]
        # add the unmetered devices to the aggregate consumption
        df["aggregate"] =  df["aggregate"] + unmetered["aggregate"]

        for c in df.columns:
            if c == "aggregate":
                continue

            df[c] = (df[c] > 5).astype(int)
        windows.append(df)
    return windows

def create_training_data(windows, labels):
    X_Y_test= []
    
    for window in tqdm(windows):
        
        x = window["aggregate"].values
        devices = [False] * len(labels)
    
        # prepare Y
        for c in window.columns:
            if c == "aggregate":
                continue
            on = (window[c] > 0)
            ix = labels.index(c)
            
            devices[ix] = on.any()

        X_Y_test.append((x, devices))

    return X_Y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process energy graph data.")
    
    # Argument for the path to the base folder
    parser.add_argument("--path_to_base", type=str, default=".", help="Path to base folder")
    
    # Argument for time window
    parser.add_argument("--time_window", type=int, default=2700, help="Time window in seconds")

    # Argument for number of windows
    parser.add_argument("--num_windows", type=int, default=100000, help="Number of windows to generate")
    
    # Argument for upper bound
    parser.add_argument("--upper_bound", type=int, default=32, help="Upper bound in seconds")
    
    # Argument for max gap
    parser.add_argument("--max_gap", type=int, default=3600, help="Max gap in seconds")

    # Argument for synthetic type
    parser.add_argument("--syn_type", type=str, default="ideal", help="Type of synthetic data to generate")
    
    args = parser.parse_args()

    # Initialize paths
    path_to_base = args.path_to_base
    path = path_to_base + "/data/processed_watts/"
    labels_path = path_to_base + "/data/labels_new.pkl"
    save_path = path_to_base + "/data/training_data/synthetic/"

    labels = pd.read_pickle(labels_path)
    
    # Initialize parameters
    time_window = args.time_window
    num_windows = args.num_windows
    upper_bound = pd.Timedelta(seconds=args.upper_bound)
    max_gap = pd.Timedelta(seconds=args.max_gap)
    
    # Print parameters
    print("Path to base folder: ", path_to_base)
    print("Path to save windows: ", save_path)
    print("Time window: ", time_window, "rows | ", time_window*8, "seconds")
    print("Number of windows: ", num_windows)
    print("Upper bound: ", upper_bound)
    print("Max gap: ", max_gap)
    print("Synthetic type: ", args.syn_type)
    print("Num devices: ", len(labels))

    # Get data and create windows
    data = get_device_windows(path, time_window, upper_bound, max_gap)

    if args.syn_type == "ideal":
        windows = generate_syn_ideal(data, num_windows, labels)

    elif args.syn_type == "unmetered":
        windows = generate_syn_unmetered(data, num_windows, labels)
    elif args.syn_type == "both":
        windows_ideal = generate_syn_ideal(data, num_windows, labels)
        windows_unmetered = generate_syn_unmetered(data, num_windows, labels)
        X_Y_ideal = create_training_data(windows_ideal, labels)
        X_Y_unmetered = create_training_data(windows_unmetered, labels)

        with open(save_path+ f"/X_Y_wsize{time_window}_numW_{num_windows}_upper{int(upper_bound.total_seconds())}_gap{int(max_gap.total_seconds())}_numD{len(labels)}_ideal.pkl", "wb") as f:
            pickle.dump(X_Y_ideal, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(save_path+ f"/X_Y_wsize{time_window}_numW_{num_windows}_upper{int(upper_bound.total_seconds())}_gap{int(max_gap.total_seconds())}_numD{len(labels)}_unmetered.pkl", "wb") as f:
            pickle.dump(X_Y_unmetered, f, protocol=pickle.HIGHEST_PROTOCOL)
        exit()

    X_Y = create_training_data(windows, labels)
    with open(save_path+ f"/X_Y_wsize{time_window}_numW_{num_windows}_upper{int(upper_bound.total_seconds())}_gap{int(max_gap.total_seconds())}_numD{len(labels)}_{args.syn_type}.pkl", "wb") as f:
        pickle.dump(X_Y, f, protocol=pickle.HIGHEST_PROTOCOL)

