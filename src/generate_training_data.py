import pandas as pd
import os
from tqdm import tqdm
import pickle
import numpy as np
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.helper import preprocess_string, generate_labels




def sample_normal_within_range(mu=7, sigma=5, a=1, b=35) -> int:
    """Sample from a normal distribution within a range of values in the given interval [a, b] with a given mean and standard deviation"""
    samples = []
    while len(samples) == 0:
        value = round(np.random.normal(mu, sigma))
        if a <= value <= b:
            samples.append(value)
    return np.array(samples)[0]


def process_dataset(dataset : str, path : Path, time_window : int, upper_bound : pd.Timedelta, max_gap: pd.Timedelta) -> dict:
    """Process the dataset and return the windows for each device
    ## Parameters
    `dataset` : Name of the dataset
    `path` : Path to the parsed data
    `time_window` : size of the window in rows
    `upper_bound` : upper bound for the gap in seconds if there is more than 15 gaps of this size in a window skip the window
    `max_gap` : max gap in seconds if there is a gap of more than this size in a window skip the window
    ## Returns
    `devices_processed_local` : Dictionary containing the list of windows for each device
    """
    devices_processed_local = {}
    if not dataset.endswith(".pkl"):
        return devices_processed_local
    data = pd.read_pickle(path / dataset)
    # iterate over households in dataset
    for house in data:
        for device in data[house]:
            # ignore aggregate as we only need to preprocess devices
            if device == "aggregate":
                continue
            # rename devices to uniform names
            name = preprocess_string(device)
            if name not in devices_processed_local:
                devices_processed_local[name] = []
            curr_device = process_data(data[house][device], time_window, upper_bound, max_gap)
            devices_processed_local[name].extend(curr_device)
    return devices_processed_local


def process_data(df: pd.DataFrame, time_window : int, upper_bound : pd.Timedelta, max_gap : pd.Timedelta) -> list:   
    """
    Process the data by resampling it to 8s and filling the gaps with the nearest value and then splitting it into windows of size time_window.
    If there is a gap of more than max_gap skip the window. If there are more than 15 gaps of upper_bound or more skip the window. If the device is always off skip the window.
    ## Parameters
    `df` : DataFrame containing the data
    `time_window` : size of the window in rows 
    `upper_bound` : upper bound for the gap in seconds if there is more than 15 gaps of this size in a window skip the window
    `max_gap` : max gap in seconds if there is a gap of more than this size in a window skip the window
    ## Returns
    `windows` : List of windows for the device
    """
    df = df.resample("8S").fillna(method="nearest", limit=4)
    df.fillna(0, inplace=True)
    # handle negatve values
    df[df < 0] = 0  # TODO FIX

    windows = []
    for i in range(0, len(df) - time_window, time_window + 1):
        window = df.iloc[i : i + time_window]

        # if there is a gap of more than max_gap skip the window
        time_diffs = window.index.to_series().diff().dropna()
        if (time_diffs >= max_gap).any():
            continue
        # if there are more than 15 gaps of upper_bound or more skip the window
        if len(time_diffs[time_diffs > upper_bound]) > 15:
            continue
        # skip if the device is always off
        if window.max().max() < 5:
            continue
        window.reset_index(drop=True, inplace=True)

        windows.append(window)

    return windows


def get_device_windows(path: Path, time_window : int, upper_bound : pd.Timedelta, max_gap : pd.Timedelta, datasets : list) -> dict:
    """
    Get the windows for each device in the dataset
    ## Parameters
    `path` : Path to the parsed data
    `time_window` : size of the window in rows
    `upper_bound` : upper bound for the gap in seconds if there is more than 15 gaps of this size in a window skip the window
    `max_gap` : max gap in seconds if there is a gap of more than this size in a window skip the window
    `datasets` : List of datasets to process
    ## Returns
    `devices_processed` : Dictionary containing the windows for each device
    """

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
            except Exception as e:  # TODO: More specific exception. This one will also keyboard interrupts
                print(f"Failed to process dataset {dataset}. Error: {e}")

    return devices_processed



# sum(devices) = aggregate
def generate_synthetic(devices_processed : dict, num_windows : int, device_list : list) -> list:
    """Generate synthetic data where the sum of the devices is equal to the aggregate consumption
    ## Parameters
    `devices_processed` : Dictionary containing the windows for each device
    `num_windows` : Number of windows to generate
    `device_list` : List of devices
    ## Returns
    `windows` : List of dataframes containing the synthetic data"""
    windows = []
    for i in tqdm(range(num_windows)):
        # get number of devices sampled from normal distribution
        nm_device = sample_normal_within_range()
        # randomly select devices from the list of devices
        selected_devices = random.sample(device_list, nm_device)

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
                selected_devices = random.sample(device_list, nm_device)
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



def create_tuples(windows : list, labels : list) -> list:
    """
    Create training data from the windows
    ## Parameters
    `windows` : List of windows
    `labels` : List of labels

    ## Returns
    `X_Y_test` : List of tuples (X, Y) where X is the aggregate consumption and Y is a list of devices on in the window
    """
    X_Y_test = []

    for window in tqdm(windows):
        x = window["aggregate"].values
        devices = [False] * len(labels)

        # prepare Y
        for c in window.columns:
            if c == "aggregate":
                continue
            on = window[c] > 0
            ix = labels.index(c)

            devices[ix] = on.any()

        X_Y_test.append((x, devices))

    return X_Y_test



def generate_training_data(data_path : Path, save_path : Path, datasets : list, window_size=2688 , num_windows=100000, upper_bound=32, max_gap=3600):
    """
    Generate training data from the parsed data and save it to a pickle file
    ## Parameters
    `data_path` : Path to the parsed data
    `save_path` : Path to save the training data
    `datasets` : List of datasets to process
    `window_size` : Size of the window in rows (8s)
    `num_windows` : Number of windows to generate
    `upper_bound` : Upper bound for the gap in seconds if there is more than 15 gaps of this size in a window skip the window
    `max_gap` : Max gap in seconds if there is a gap of more than this size in a window skip the window

    
    """
    # add .pkl to datasets if not present
    for i in range(len(datasets)):
        if not datasets[i].endswith(".pkl"):
            datasets[i] = datasets[i] + ".pkl"
    
    

    upper_bound = pd.Timedelta(seconds=upper_bound)
    max_gap = pd.Timedelta(seconds=max_gap)
    # generate labels from the parsed data and save to a pickle file
    labels = generate_labels(data_path, save_path, datasets)
    data = get_device_windows(data_path, window_size, upper_bound, max_gap, datasets)

    windows = generate_synthetic(data, num_windows, labels)

    X_Y = create_tuples(windows, labels)
    print("Saving data to: ", save_path / f"X_Y_wsize{window_size}_numW_{num_windows}_upper{int(upper_bound.total_seconds())}_gap{int(max_gap.total_seconds())}_numD{len(labels)}.pkl")
    with open(save_path / f"X_Y_wsize{window_size}_numW_{num_windows}_upper{int(upper_bound.total_seconds())}_gap{int(max_gap.total_seconds())}_numD{len(labels)}.pkl", "wb") as f:
        pickle.dump(X_Y, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process energy graph data.")

    # Argument for the path to the base folder
    parser.add_argument("--data_path", type=str, default="./data/parsed/", help="Path to base folder (default: `.`)")

    # Argument for the path to the save folder
    parser.add_argument("--save_path", type=str, default="./data/training_data/synthetic/", help="Path to save folder (default: `.`)")

    # Argument for time window
    parser.add_argument("--window_size", type=int, default=2688, help="window size (default: 2688)")

    # Argument for number of windows
    parser.add_argument("--num-windows", type=int, default=100_000, help="Number of windows to generate (default: 100000)")

    # Argument for upper bound
    parser.add_argument("--upper-bound", type=int, default=32, help="Upper bound in seconds (default: 32)")

    # Argument for max gap
    parser.add_argument("--max-gap", type=int, default=3600, help="Max gap in seconds (default: 3600)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)


    # Initialize paths
    data_path : Path = Path(args.data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    save_path : Path = Path(args.save_path).resolve()
    assert save_path.exists(), f"Path '{save_path}' does not exist!"



    # Initialize parameters
    time_window = args.window_size
    num_windows = args.num_windows
    upper_bound = pd.Timedelta(seconds=args.upper_bound)
    max_gap = pd.Timedelta(seconds=args.max_gap)

    generate_training_data(data_path, save_path, time_window, num_windows, upper_bound, max_gap)


