from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from src.helper import save_to_pickle
######################DATASET INFO#########################################
# sampling rate: 15min
# length: 1.8 years
# unit: kWh
# households: 110953
# no submeter data
# Location: Uruguay
# Source: https://www.nature.com/articles/s41597-022-01122-x

# DATA_PATH = ""


def process_file(file_path: str) -> dict:
    # file_path = os.path.join(DATA_PATH, 'consumption_data', file)

    file_path: Path = Path(file_path).resolve()
    assert file_path.exists(), f"Path '{file_path}' does not exist!"

    df = pd.read_csv(file_path)
    # pivot the dataframe so that each column is a different house with timestamps as the index and the values are the consumption
    df = df.pivot(index="datetime", columns="id", values="value")
    # convert the timestamps to datetime objects and set the correct timezone
    df.index = pd.to_datetime(df.index, unit="s", utc=True).tz_convert("America/Montevideo")
    df.sort_index(inplace=True)

    temp_data = defaultdict(lambda: {"aggregate": []})
    # iterate over each column and add the data to the dictionary and drop missing values
    for col in df.columns:
        name = "ECDUY_" + str(col)
        temp_data[name]["aggregate"].append(df[col].dropna())

    return dict(temp_data)


def parse_ECDUY(data_path: str, save_path: str, batch_size: int = 6, n_jobs: int = 32) -> None:
    n_jobs = int(os.cpu_count() // 8)
    if os.cpu_count() < 16:
        n_jobs = os.cpu_count() // 2

    # no need for more than 32 processes for this dataset
    if n_jobs > 32:
        n_jobs = 32
    data_path: Path = Path(data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    # set the global data path variable
    global DATA_PATH
    DATA_PATH = data_path

    # get all the files
    files = [f for f in os.listdir(os.path.join(data_path, "consumption_data")) if f.endswith(".csv")]

    # Fix filenames to full path
    files = [os.path.join(data_path, "consumption_data", filename) for filename in files]

    # batch_size = 11  # or whatever size you deem fit, based on your system's number of cores and memory there is 22 file total for size 11 around 500gb is

    data = defaultdict(lambda: {"aggregate": []})

    # process the batches
    for i in tqdm(range(0, len(files), batch_size)):
        batch_files = files[i : i + batch_size]
        # use a process pool to parallelize the processing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_file, batch_files))
        # merge the results
        for result in results:
            for key, value in result.items():
                data[key]["aggregate"].extend(value["aggregate"])

    # Convert defaultdict back to a normal dictionary
    data = dict(data)

    # merge the dataframes
    for key in tqdm(data):
        data[key]["aggregate"] = pd.concat(data[key]["aggregate"]).sort_index()

    # save the data
    save_to_pickle(data, save_path)
