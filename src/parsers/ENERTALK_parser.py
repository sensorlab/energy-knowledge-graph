import os
import pandas as pd
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
from src.helper import save_to_pickle
import multiprocessing
from pathlib import Path


######################DATASET INFO#########################################
# sampling rate: 1s
# length: 4 months
# unit: watts
# households: 22
# submetered: yes
# Location: South Korea
# Source: https://www.nature.com/articles/s41597-019-0212-5


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["reactive_power"], inplace=True)
    # convert unix timestamp to datetime and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC", ambiguous="infer").dt.tz_convert("Asia/Seoul")

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # handle duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # resample to 1 second
    df = df.resample("1S").sum()

    return df


def parse_name(file_name: str) -> str:
    """
    Parse the name of the file to get the device name"
    """
    # remove the extension
    file_name = file_name.split(".")[0]
    # get the device name
    file_name = file_name.split("_")[1]

    if file_name == "total":
        file_name = "aggregate"

    return file_name


def process_house(house_path, queue) -> tuple[str, dict]:
    house_path: Path = Path(house_path).resolve()
    assert house_path.exists(), f"Path '{house_path}' does not exist!"
    house = os.path.basename(house_path)  # Extract house name from the path
    house_dict = defaultdict(list)
    house_name = "ENERTALK_" + str(int(house))

    for day in os.listdir(house_path):
        day_path = os.path.join(house_path, day)
        for device in os.listdir(day_path):
            device_path = os.path.join(day_path, device)
            name = parse_name(device)
            df = preprocess_dataframe(pd.read_parquet(device_path))
            house_dict[name].append(df)

    for key in house_dict:
        df = pd.concat(house_dict[key], axis=0)
        # remove duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]
        house_dict[key] = df

    queue.put(1)  # Indicate that one house has been processed
    return house_name, house_dict


def parse_ENERTALK(data_path: str, save_path: str) -> None:
    data_dict = {}
    house_paths = [os.path.join(data_path, house) for house in os.listdir(data_path)]
    queue = multiprocessing.Manager().Queue()
    
    # there are only 22 households no need to use more than 4 cores
    cpu_count = 4
    if os.cpu_count() < 4:
        cpu_count = 1
    with tqdm(total=len(house_paths), desc="Processing houses", unit="house") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            houses = [executor.submit(process_house, house_path, queue) for house_path in house_paths]

            # Update progress bar based on queue
            for _ in concurrent.futures.as_completed(houses):
                progress_bar.update(queue.get())

            for house in houses:
                house_name, house_dict = house.result()
                data_dict[house_name] = house_dict

    save_to_pickle(data_dict, save_path)
