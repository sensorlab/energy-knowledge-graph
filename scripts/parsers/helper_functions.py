import pandas as pd
import pickle

# watts to kWh given data frequency as a fraction of an hour (e.g. 0.5 for half-hourly data)
def watts2kwh(df, data_frequency):
    df = df/1000 * data_frequency
    return df

# save a dictionary to a pickle file
def save_to_pickle(dict : dict, filename : str):
    try:
        with open(filename, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data successfully saved to ", filename)
    except Exception as e:
        print("Failed to save data: ", e)
