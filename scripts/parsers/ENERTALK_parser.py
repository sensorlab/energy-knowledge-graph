import os
import pandas as pd
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
from helper_functions import save_to_pickle, watts2kwh
import multiprocessing





def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the name of the file to get the device name"
    """
    df.drop(columns=["reactive_power"], inplace=True)
    # convert unix timestamp to datetime and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    df.set_index("timestamp", inplace=True)
    # convert to kWh 15hz data
    df = watts2kwh(df, (1/15)/3600)
    # resample to 1 second
    df = df.resample("1S").sum()

    return df

def parse_name(file_name: str):
    """
    Parse the name of the file to get the device name"
    """
    # remove the extension
    file_name = file_name.split(".")[0]
    # get the device name
    file_name = file_name.split("_")[1]
 

    return file_name


def process_house(house_path, queue):
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
        house_dict[key] = pd.concat(house_dict[key], axis=0)

    queue.put(1)  # Indicate that one house has been processed
    return house_name, house_dict



def parse_ENERTALK(data_path, save_path):

    data_dict = {}
    # data_path = "./Energy_graph/data/temp/ENERTALK/"
    house_paths = [os.path.join(data_path, house) for house in os.listdir(data_path)]
    queue = multiprocessing.Manager().Queue()

    with tqdm(total=len(house_paths), desc="Processing houses", unit="house") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()/2) as executor:
            futures = [executor.submit(process_house, house_path, queue) for house_path in house_paths]
            
            # Update progress bar based on queue
            for _ in concurrent.futures.as_completed(futures):
                progress_bar.update(queue.get())

            for future in futures:
                house_name, house_dict = future.result()
                data_dict[house_name] = house_dict




    save_to_pickle(data_dict, save_path)