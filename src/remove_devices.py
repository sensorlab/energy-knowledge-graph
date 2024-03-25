import pandas as pd
import pickle
from pathlib import Path
import shutil

def HES_remove(read_path, save_path):
    print("HES: ")
    df = pd.read_pickle(read_path/"HES.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "sockets" in k or "light" in k or "heat" in k or "outlet" in k or "lamp" in k:
                print("REMOVE: ", k)
                del df[h][k]
            
    
    df["HES_1"].keys()


    # save to pickle
    with open(save_path/"HES.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")

def REDD_remove(read_path, save_path):
    print("REDD: ")
    df = pd.read_pickle(read_path/"REDD.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "sockets" in k or "light" in k or "CE appliance" in k or "subpanel" in k or "air handling unit" in k:
                print("REMOVE: ", k)
                del df[h][k]
            # print(k)
    
    # df["HES_1"].keys()


    # save to pickle
    with open(save_path/"REDD.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")


def ECO_remove(read_path, save_path):
    print("ECO: ")
    df = pd.read_pickle(read_path/"ECO.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "Lamp" in k or "light" in k or "CE appliance" in k or "subpanel" in k or "Tablet" in k:
                print("REMOVE: ", k)
                del df[h][k]
  

    # save to pickle
    with open(save_path/"ECO.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")

def UKDALE_remove(read_path, save_path):
    print("UKDALE: ")
    df = pd.read_pickle(read_path/"UKDALE.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "set top box" in k or "light" in k or "plug" in k or "lamp" in k  or "charger" in k or "USB" in k or "radio" in k or "baby" in k or "bouncy castle pump" in k or "solar thermal pumping station" in k or "external hard disk" in k:
                print("REMOVE: ", k)
                del df[h][k]
        


    # save to pickle
    with open(save_path/"UKDALE.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")

def REFIT_remove(read_path, save_path):
    print("REFIT: ")
    df = pd.read_pickle(read_path/"REFIT.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "kettle/toaster" in k or "dehumidifier/heater" in k or "plug" in k or "lamp" in k or "vivarium" in k or "oven extractor fan" in k:
                print("REMOVE: ", k)
                del df[h][k]


    # save to pickle
    with open(save_path/"REFIT.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")

def SUST2_remove(read_path, save_path):
    print("SUST2: ")
    df = pd.read_pickle(read_path/"SUST2.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "HairDryer-Straightener" in k or "dehumidifier/heater" in k or "plug" in k or "lamp" in k:
                print("REMOVE: ", k)
                del df[h][k]

    # save to pickle
    with open(save_path/"SUST2.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")

def DEDDIAG_remove(read_path, save_path):
    print("DEDDIAG: ")
    df = pd.read_pickle(read_path/"DEDDIAG.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "Office Desk" in k or "dehumidifier/heater" in k or "plug" in k or "lamp" in k:
                print("REMOVE: ", k)
                del df[h][k]


    # save to pickle
    with open(save_path/"DEDDIAG.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")

def HEART_remove(read_path, save_path):
    print("HEART: ")
    df = pd.read_pickle(read_path/"HEART.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "Office Desk" in k or "dehumidifier/heater" in k or "plug" in k or "lamp" in k or "radio" in k:
                print("REMOVE: ", k)
                del df[h][k]

    # save to pickle
    with open(save_path/"HEART.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")

def IAWE_remove(read_path, save_path):
    print("IAWE:")
    df = pd.read_pickle(read_path/"IAWE.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "wet appliance" in k or "dehumidifier/heater" in k or "plug" in k or "lamp" in k or "motor" in k:
                print("REMOVE: ", k)
                del df[h][k]

    # save to pickle
    with open(save_path/"IAWE.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("*************************************************************")

def DRED_remove(read_path, save_path):
    print("DRED:")
    df = pd.read_pickle(read_path/"DRED.pkl")
    for h in df:
        keys = list(df[h].keys())
        for k in keys:
            if "sockets" in k or "dehumidifier/heater" in k or "plug" in k or "lamp" in k:
                print("REMOVE: ", k)
                del df[h][k]

    # save to pickle
    with open(save_path/"DRED.pkl", 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*************************************************************")



def remove_devices(read_path : Path, save_path : Path, datasets : list[str]):
    """
    Remove devices from the datasets that we ignore during training

    ## Parameters:
    `read_path` : The path to the folder with the raw data
    `save_path` : The path to the folder to save the cleaned data
    `datasets` : The list of datasets to remove devices from
    
    """
    functions = {
    "HES" : HES_remove,
    "REDD" : REDD_remove,
    "ECO" : ECO_remove,
    "UKDALE" : UKDALE_remove,
    "REFIT" : REFIT_remove,
    "DEDDIAG" : DEDDIAG_remove,
    "HEART" : HEART_remove,
    "IAWE" : IAWE_remove,
    "DRED" : DRED_remove
    }

    for dataset in datasets:
        # check if the dataset is in the functions
        if dataset in functions:
            functions.get(dataset)(read_path, save_path)
        else:
            shutil.copy2(read_path / (dataset+".pkl"), save_path / (dataset+ ".pkl"))

            

