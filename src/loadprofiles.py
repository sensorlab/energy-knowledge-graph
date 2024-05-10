import argparse
import concurrent.futures
import os
import pickle
from pathlib import Path
from .helper import watts2kwh

import pandas as pd
from tqdm import tqdm



def calculate_loadprofiles(df: pd.DataFrame) -> dict:
    """
        Calculate the daily, weekly and monthly load profiles of a given device or household
        ### Parameters
        `df` : should be of the form [datetime index, value in watts]
    """
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


def process_dataset(dataset: str, data_path: Path) -> dict:
    """
    Process a dataset and return the load profiles as a dictionary
    ### Parameters
    `dataset` : the name of the dataset
    `data_path` : Path to the folder containing parsed datasets

    """
    data_dict = pd.read_pickle(os.path.join(data_path, dataset))
    loadprofiles = {}
    for house in data_dict:
        house_lp = {}
        for device in data_dict[house]:
            # these datasets are already in kWh
            if "ECDUY" in house or "DEKN" in house or "HUE" in house:
                house_lp[device] = calculate_loadprofiles(data_dict[house][device])
            else:
                # get sampling rate for kWh conversion
                time_deltas = data_dict[house][device].index.to_series().diff().dropna()
                median_time_delta = time_deltas.median()
                sampling_rate = median_time_delta.total_seconds() / 3600
                house_lp[device] = calculate_loadprofiles(watts2kwh(data_dict[house][device], sampling_rate))

        loadprofiles[house] = house_lp

    return loadprofiles


def generate_loadprofiles(data_path: Path, save_path: Path, datasets: list[str]) -> None:
    """
    Generate load profiles for a list of datasets and save to a pickle file
    ### Parameters
    `data_path` : Path to the folder containing parsed datasets
    `save_path` : Path to the folder to save the load profiles
    `datasets` : List of datasets to process as a list of strings containing the dataset names
    """

    dataset_paths = [dataset for dataset in os.listdir(data_path) if
                     dataset.endswith('.pkl') and (dataset.split(".")[0] in datasets)]

    cpu_count = int(os.cpu_count() // 4)

    cpu_count = len(datasets) if cpu_count > len(datasets) else cpu_count
    data_dict = {}
    with tqdm(total=len(dataset_paths), desc="Processing datasets", unit="dataset") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(process_dataset, dataset, data_path) for dataset in dataset_paths]

            for future in concurrent.futures.as_completed(futures):
                dataset_loadprofile = future.result()
                data_dict.update(dataset_loadprofile)

                progress_bar.update(1)  # update progress bar

    # combined file containing all loadprofiles
    with open(save_path / "merged_loadprofiles.pkl", 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

