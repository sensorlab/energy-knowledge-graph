import argparse
import concurrent.futures
import os
import pickle
from collections import Counter

import pandas as pd
from tqdm import tqdm

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
    df = data["aggregate"].resample("8s").mean()
    df.columns = ["aggregate"]
    dfs.append(df)
    for device in data:
        # ignore devices
        if any(ignored_device in device.lower() for ignored_device in ignored_devices):
            continue
        if device == "aggregate":
            continue
        # preprocess device name
        device_name = preprocess_string(device)

        df = data[device]
        df = df.resample("8s").mean()

        # rename column to standardized device name
        df.columns = [device_name]
        if df.max().max() < 2:
            print("device with zeros: ", device_name)
            continue

        time_diffs = df.index.to_series().diff()
        median_interval = time_diffs.median()

        # if there is less than 3 days of data drop the device
        if len(df) < (3 * 24 * 60 * 60) / median_interval.total_seconds():
            print("less than 3 days of data for device: ", device_name)
            continue
        df.dropna(inplace=True)
        dfs.append(df)

    # concatenate all dataframes
    df = pd.concat(dfs, axis=1)

    # handle missing values
    df = df.ffill(limit=6)
    df.fillna(0, inplace=True)

    # handle negative values
    df[df < 0] = 0

    df["sum_ideal"] = df.sum(axis=1) - df["aggregate"]
    df.drop(columns=["aggregate"], inplace=True)

    df.rename(columns={"sum_ideal": "aggregate"}, inplace=True)

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


def get_data(path: str, labels_path: str, values=0):
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


def create_windows(data: dict, labels_path: str, save_path: str, time_window=2700, upper_bound=pd.Timedelta(seconds=32),
                   max_gap=pd.Timedelta(seconds=3600)):
    """Creates windows of time_window seconds from the data and discards windows with gaps of more than 1h or 15 gaps of 32 seconds or more"""
    labels = pd.read_pickle(labels_path)
    # windows = []
    X_Y = []  # list of tuples (X, Y)
    skip_count_1 = 0
    skip_count_2 = 0
    skip_count_3 = 0
    total_count = 0

    for df in tqdm(data.values()):
        for i in range(0, len(df) - time_window, time_window + 1):
            window = df.iloc[i:i + time_window]
            total_count += 1
            # if there is a gap of more than max_gap skip the window
            time_diffs = window.index.to_series().diff().dropna()
            if (time_diffs >= max_gap).any():
                skip_count_1 += 1
                continue
            # if there are more than 15 gaps of upper_bound or more skip the window
            if len(time_diffs[time_diffs > upper_bound]) > 15:
                skip_count_2 += 1
                continue

            x = window["aggregate"].values
            # if there is a value bigger than 50000 skip the window
            if (x > 50000).any():
                skip_count_3 += 1
                continue
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

    print("Total windows: ", total_count, "Skipped windows due to 30min gap: ", skip_count_1,
          "Skipped windows due to 15 gaps of 32s or more: ", skip_count_2,
          "Skipped windows due to values larger than 50k: ", skip_count_3, "Procentage skipped: ",
          (skip_count_1 + skip_count_2 + skip_count_3) / total_count * 100)
    with open(
            save_path + f"/X_Y_wsize{time_window}_upper{int(upper_bound.total_seconds())}_gap{int(max_gap.total_seconds())}_numD{len(labels)}.pkl",
            "wb") as f:
        pickle.dump(X_Y, f, protocol=pickle.HIGHEST_PROTOCOL)


