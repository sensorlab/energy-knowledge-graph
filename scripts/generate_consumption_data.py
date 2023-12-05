import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
import argparse
import concurrent.futures
def average_daily_consumption(df: pd.DataFrame):
    """Returns the average daily consumption of a device in kWh."""
    df = df.copy()
    # convert to kWh
    time_deltas = df.index.to_series().diff().dropna()
    median_time_delta = time_deltas.median()
    sampling_rate = median_time_delta.total_seconds()/3600
    df/=1000
    df *= sampling_rate
    # resample to daily
    df = df.resample('1D').sum()

    # check if dataframe is empty and return 0 if it is
    if df.empty:
        return 0
    # return mean of daily consumption
    return df.values.mean()    
def average_on_off_event(df: pd.DataFrame):
    """Returns the average on and off event of a device in kWh."""
    df_orig = df.copy()
    time_deltas = df.index.to_series().diff().dropna()
    median_time_delta = time_deltas.median()
    sampling_rate = median_time_delta.total_seconds()/3600

    threshold = 5  # Define the threshold for 'on' state
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
        start_index = max(start_index , 0)
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
        start_index = max(start_index , 0)
        end_index = min(end_index , len(df) - 1)

        merged_on_period = df.iloc[start_index:end_index + 1]
        merged_on_periods.append(merged_on_period)

    avg = []
    for p in merged_on_periods:
        p = p.iloc[:,0].copy()
        p/=1000
        p *= sampling_rate
        
        # break
        avg.append(p.sum())

    return np.array(avg).mean()
    
# Function to process each dataset
def process_dataset(dataset_path):

    house_data_dict = {}
    data = pd.read_pickle(dataset_path)
    for h in data.keys():
        house_data = {}
        df = data[h]
        # for aggregate data calculate only daily consumption
        for d in df.keys():
            if "aggregate" in d:
                house_data[d] = {"daily": average_daily_consumption(df[d])}
            else:
                house_data[d] = {"daily": average_daily_consumption(df[d]), "event": average_on_off_event(df[d])}
        house_data_dict[h] = house_data
    del data
    return house_data_dict

def main(DATA_PATH, SAVE_PATH):

    # limit to half of cpu cores
    cpu_count = int(os.cpu_count() / 2)
    
    # get all dataset paths
    dataset_paths = [os.path.join(DATA_PATH, d) for d in os.listdir(DATA_PATH)]

    # process each dataset in parallel
    consumption_data = {}
    with tqdm(total=len(dataset_paths), desc="Processing datasets", unit="dataset") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {executor.submit(process_dataset, dataset_path): dataset_path for dataset_path in dataset_paths}
            
            for future in concurrent.futures.as_completed(futures):
                house_data_dict = future.result()
                consumption_data.update(house_data_dict)
                progress_bar.update(1)

    

    # save consumption data
    with open(os.path.join(SAVE_PATH, "consumption_data.pkl"), 'wb') as f:
        pickle.dump(consumption_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate consumption data.')

    parser.add_argument('--data_path', type=str, help='Path to data folder.', default="../data/watts_test/")

    parser.add_argument('--save_path', type=str, help='Path to save folder.', default="../data/metadata/")

    args = parser.parse_args()

    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path
    main(DATA_PATH, SAVE_PATH)