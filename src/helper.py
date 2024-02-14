import re
import numpy as np
import pandas as pd
import pickle

def preprocess_string(string : str) -> str:
    string = string.lower().strip()
    string = re.sub(' +', ' ', string)
    string = string.replace("_", " ")
    string = string.replace("-", " ")
    string = string.replace("&", " ")
    string = string.split("(")[0]
    string = string.split("#")[0]

    string = string.strip()

    # handle known synoynms
    synonyms = {
        "refrigerator": "fridge",
        "vaccumcleaner": "vacuum cleaner",
        "breadmaker": "bread maker",
      
        
    }
    if "freezer" in string:
        string = "fridge"

    if string in synonyms:
        string = synonyms[string]

    if 'hi fi' in string:
        string = "audio system"

    if "router" in string:
        string = "router"

    if "treadmill" in string:
        string = "running machine"
        

    if "laptop" in string:
        string = "laptop"
    
    if "server" in string:
        string = "server"

    if "monitor" in string and not "baby" in string:
        string = "monitor"
    # special cases
    if "computer" in string and "charger" not in string:
        string = "pc"

    if "tv" in string:
        string = "television"

    if "television" in string:
        string = "television"

    if "macbook" in string:
        string = "laptop"
        
    if "car charger" == string:
        string = "ev"
    
    if "toast" in string:
        string = "toaster"
    
    if "modem" in string:
        string = "router"

    # we treat all audio devices as speakers so subwoofer is also a speaker
    if "subwoofer" in string:
        string = "speaker"

    if "speaker" in string:
        string = "speaker"

    if "iron" in string and "soldering" not in string:
        string = "iron"

    
    if "coffeemachine" in string:
        string = "coffee machine"
    if "coffee maker" in string:
        string = "coffee machine"

    if "dishwasher" in string:
        string = "dish washer"
    if "air conditioner" in string:
        string = "ac"

    if "air conditioning" in string:
        string = "ac"
    
    string = re.sub(' +', ' ', string)
    string = re.sub(r'\d+', '', string)
    return string.strip()


# min-max normalization Xmin=0 
def normalize(X):
    max_value = 0

    for x in X:
        v = np.max(x)
        if v > max_value:
            max_value = v

    if max_value == 0:
        return X
    return X / max_value


# watts to kWh given data frequency as a fraction of an hour (e.g. 0.5 for half-hourly data)
def watts2kwh(df: pd.Series, data_frequency: float) -> pd.Series:
    df = df / 1000 * data_frequency
    return df


# save a dictionary to a pickle file
def save_to_pickle(dict: dict, filename: str):
    try:
        with open(filename, "wb") as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data successfully saved to ", filename)
    except Exception as e:
        print("Failed to save data: ", e)
