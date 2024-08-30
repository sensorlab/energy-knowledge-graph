import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
import argparse
import concurrent.futures
from pathlib import Path


def average_daily_consumption(df: pd.DataFrame, kWh=False) -> float:
    """
    Calculate the average daily consumption of a device in kWh
    ### Parameters
    `df` : should be in the form datetime index and column should contain device consumption readings in watts
    `kWh` : if True, the data is already in kWh
    ### Returns
    `float` : average daily consumption in kWh
    """

    if df.empty:
        return 0
    df = df.copy()
    if not kWh:
        time_deltas = df.index.to_series().diff().dropna()
        median_time_delta = time_deltas.median()
        sampling_rate = median_time_delta.total_seconds() / 3600
        df /= 1000
        df *= sampling_rate
    df = df.resample('1D').sum()

    return df.values.mean()


# noinspection PyBroadException
def average_on_off_event(df: pd.DataFrame, kWh=False) -> float:
    """
    Calculate the average on/off event consumption of a device in kWh
    ### Parameters
    `df` : should be in the form datetime index and column should contain device consumption readings in watts
    `kWh` : if True, the data is already in kWh
    ### Returns
    `float` : average on/off event consumption in kWh
    """
    # check if df is empty
    if df.empty:
        return 0
    

    df_orig = df.copy()
    time_deltas = df.index.to_series().diff().dropna()
    median_time_delta = time_deltas.median()
    sampling_rate = median_time_delta.total_seconds() / 3600

    # if data is in kWh, convert to watts
    if kWh:
        df /= sampling_rate
        df *= 1000



    threshold = 5  # Define the threshold for 'on' state
    # noinspection PyUnresolvedReferences
    df['above_threshold'] = (df >= threshold).astype(int)
    df['rolling_sum'] = df['above_threshold'].rolling(window=6, min_periods=1).sum()

    # Define 'on' state as being above threshold for more than 5 consecutive rows
    df['state'] = (df['rolling_sum'] > 5).astype(int)
    df['state_change'] = df['state'].diff().fillna(0)

    # Find start and end times of 'on' events
    on_starts = df_orig.index[df['state_change'] == 1].tolist()
    on_ends = df_orig.index[df['state_change'] == -1].tolist()

    # Adjust for edge cases (e.g., if the series starts or ends with an 'on' state)

    if df['state'].iloc[0] == 1:
        on_starts.insert(0, df.index[0])
    if df['state'].iloc[-1] == 1:
        on_ends.append(df.index[-1])

    # Extract 'on' periods with additional rows
    on_periods = []
    for start, end in zip(on_starts, on_ends):
        start_index = df.index.get_loc(start)
        end_index = df.index.get_loc(end)

        # Adjust start and end index to add additional rows
        start_index = max(start_index, 0)
        end_index = min(end_index, len(df) - 1)

        on_period = df.iloc[start_index:end_index + 1]
        on_periods.append(on_period)

    try:
        merged_on_starts = [on_starts[0]]
        merged_on_ends = [on_ends[0]]  # Initialize with the first 'end' event
    except:
        return 0

    for i in range(1, len(on_starts)):
        start_index = df.index.get_loc(on_starts[i])
        end_index = df.index.get_loc(merged_on_ends[-1])

        # If the gap between the end of the previous event and the start of the current event is less than 10 rows
        if start_index - end_index < 10:
            continue  # Skip this start, as it's part of the ongoing event
        else:
            merged_on_ends.append(on_ends[i - 1])
            merged_on_starts.append(on_starts[i])

    # Ensure the last end is added
    if df['state'].iloc[-1] == 1:
        merged_on_ends[-1] = on_ends[-1]
    else:
        merged_on_ends.append(on_ends[-1])

    # Extract 'on' periods with additional rows considering merged events
    merged_on_periods = []
    for start, end in zip(merged_on_starts, merged_on_ends):
        start_index = df.index.get_loc(start)
        end_index = df.index.get_loc(end)

        # Adjust start and end index to add additional rows
        start_index = max(start_index, 0)
        end_index = min(end_index, len(df) - 1)

        merged_on_period = df.iloc[start_index:end_index + 1]
        merged_on_periods.append(merged_on_period)

    avg = []
    for p in merged_on_periods:
        p = p.iloc[:, 0].copy()
    
        p /= 1000
        p *= sampling_rate

        # break
        avg.append(p.sum())

    return np.array(avg).mean()


# Function to process each dataset
def process_dataset(dataset_path):
    """
    Process a dataset and return the consumption data
    ### Parameters
    `dataset_path` : Path to the dataset
    ### Returns
    `dict` : A dictionary containing the consumption data for each house in the dataset
    """

    house_data_dict = {}
    data = pd.read_pickle(dataset_path)

    for h in data.keys():
        house_data = {}
        df = data[h]
        if "ECDUY" in h or "DEKN" in h or "HUE" in h:
            for d in df.keys():
                # for aggregate process only daily consumption
                if "aggregate" in d:
                    house_data[d] = {"daily": average_daily_consumption(df[d].copy(), kWh=True)}
                else:
                    house_data[d] = {"daily": average_daily_consumption(df[d].copy(), kWh=True),
                                    "event": average_on_off_event(df[d].copy(), kWh=True)}
        else:
            for d in df.keys():
                # for aggregate process only daily consumption
                if "aggregate" in d:
                    house_data[d] = {"daily": average_daily_consumption(df[d].copy())}
                else:
                    house_data[d] = {"daily": average_daily_consumption(df[d].copy()),
                                    "event": average_on_off_event(df[d].copy())}


        house_data_dict[h] = house_data
    del data
    return house_data_dict


# noinspection PyShadowingNames
def generate_consumption_data(data_path: Path, save_path: Path, datasets: list[str]) -> None:
    """
    Generate consumption data for a list of datasets and save to a pickle file
    ### Parameters
    `data_path` : Path to the folder containing parsed datasets
    `save_path` : Path to the folder to save the consumption data
    `datasets` : List of datasets to process example: ["REFIT", "ECO"] will process only REFIT and ECO
    """
    if os.cpu_count() < len(datasets):
        cpu_count = os.cpu_count() // 2

    else:
        cpu_count = len(datasets)

    cpu_count = int(cpu_count)
    # get all dataset paths
    dataset_paths = []
    for d in os.listdir(data_path):
        if d.split(".")[0] not in datasets:
            continue
        if d.endswith(".pkl"):
            dataset_paths.append(os.path.join(data_path, d))

    consumption_data = {}
    # process each dataset in parallel and save results to dictionary
    with tqdm(total=len(dataset_paths), desc="Processing datasets", unit="dataset") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {executor.submit(process_dataset, dataset_path): dataset_path for dataset_path in dataset_paths}

            for future in concurrent.futures.as_completed(futures):
                house_data_dict = future.result()
                consumption_data.update(house_data_dict)
                progress_bar.update(1)

    with open(os.path.join(save_path, "consumption_data.pkl"), 'wb') as f:
        pickle.dump(consumption_data, f, pickle.HIGHEST_PROTOCOL)


