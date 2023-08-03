import pandas as pd
import os
import numpy as np
from datetime import date
import yaml
import sys
import argparse

DATA_PATH = "./energy-knowledge-graph/data/metadata/datasets/"
SAVE_PATH = "./energy-knowledge-graph/data/metadata/"

def HUE_metadata():
    # read data
    HUE_metadata = pd.read_parquet(DATA_PATH+"HUE_metadata.parquet")

    # add name column
    HUE_metadata["name"] = "HUE_" + HUE_metadata["residential_id"].astype(str)


    # reset index and drop unnecessary columns
    HUE_metadata.reset_index(drop=True, inplace=True)
    HUE_metadata.drop(columns=["residential_id", "region", "tz"], inplace=True)

    # convert datetime to date only
    HUE_metadata["first_reading"] = HUE_metadata["first_reading"].dt.date
    HUE_metadata["last_reading"] = HUE_metadata["last_reading"].dt.date

    # move name to start of df
    col = HUE_metadata.pop("name")
    HUE_metadata.insert(0,"name", col)

    # rename columns
    HUE_metadata.rename(columns={"RUs": "rental_units"}, inplace=True)

    # encode AC and heating
    HUE_metadata["AC"] = 1 - HUE_metadata["NAC"]
    HUE_metadata["heating"] = np.where(HUE_metadata['GEOTH'] == 1, "geothermal", 'natural gas')

    # drop unnecessary columns
    HUE_metadata = HUE_metadata.drop(columns=['SN', 'FAGF', 'HP', 'FPG', 'FPE', 'IFRHG', 'NAC', 'FAC', 'PAC',
       'BHE', 'IFRHE', 'WRHIR', 'GEOTH'])
    
    return HUE_metadata
    
def REFIT_metadata():
    REFIT_metadata = pd.read_parquet(DATA_PATH+"refit_metadata.parquet")

    # drop unnecessary columns and add name column
    REFIT_metadata.drop(columns=["tz", "location"], inplace=True)
    REFIT_metadata['name'] = 'REFIT_' + REFIT_metadata['house'].astype(str)


    # reset index and drop unnecessary columns
    REFIT_metadata.reset_index(drop=True, inplace=True)
    REFIT_metadata.drop(columns=["house","appliances"], inplace=True)

    # move name to start of df
    col = REFIT_metadata.pop("name")
    REFIT_metadata.insert(0, col.name, col)

    # simplify house_type and change country code to country name
    REFIT_metadata["house_type"] = REFIT_metadata["house_type"].replace(" Detached   ", "house")
    REFIT_metadata["country"] = REFIT_metadata["country"].replace("GB", "United Kingdom")


    # read actual data for first and last reading
    # TODO change path
    data = pd.read_pickle(DATA_PATH+"REFIT.pkl")
    data.keys()

    # get first and last reading for each house
    start_end = {}
    for house in data.keys():
        # print(house)
        start_end[house] = {}
        start_end[house]['first_reading'] = data[house]["aggregate"].index.min().date()
        start_end[house]['last_reading'] = data[house]["aggregate"].index.max().date()

    # add first and last reading to metadata
    first_readings = [start_end[h]["first_reading"] for h in start_end]
    last_readings = [start_end[h]["last_reading"] for h in start_end]
    REFIT_metadata["first_reading"] = first_readings
    REFIT_metadata["last_reading"] = last_readings
    # drop appliances_owned column as the data is already present in the devices table
    REFIT_metadata.drop(columns=["appliances_owned"], inplace=True)

    return REFIT_metadata

def UCIML_metadata():
    data_uciml = pd.read_parquet(DATA_PATH+"uciml_household.parquet")
    # 2006-12-16
    # drop unnecessary columns
    data_uciml.drop(columns=["global_active_power", "global_reactive_power", "voltage", "global_intensity", "sub_metering_1", "sub_metering_2", "sub_metering_3", "unmetered"], inplace=True)

    # meta data for uciml
    # get first and last reading data
    first_reading = data_uciml["timestamp"].min().date()
    last_reading = data_uciml["timestamp"].max().date()

    # get country, lat and lon
    country = data_uciml["country"].iloc[0]
    lat = data_uciml["lat"].iloc[0]
    lon = data_uciml["lon"].iloc[0]
    
    # store data in a dictionary and convert to dataframe
    data = {
        "name" : "UCIML_1",
        "first_reading" :first_reading,
        "last_reading" :last_reading,
        "house_type" : "house",
        "country" :country,
        "lat" :lat,
        "lon" :lon,
        }
    
    UCIML_metadata = pd.DataFrame(data, index=[0])

    return UCIML_metadata

def HES_metadata():
    # data from https://github.com/ETSSmartRes/HES-Dataset

    data = {
        "name" : "HES_1",
        "first_reading" : date(2018, 5, 12),
        "last_reading" : date(2018,10, 10),
        "lat": 	45.508888,
        "lon": -73.561668,
        "house_type": "house",
        "country": "Canada",
    }

    HES_meta = pd.DataFrame(data, index=[0])
    return HES_meta

def ECO_metadata():
    # data from
    houses = {
    'ECO_1': {
        'first_reading': date(2012, 6, 1),
        'last_reading': date(2013, 1, 31),
        'country': 'Switzerland'
    },
    'ECO_2': {
        'first_reading': date(2012, 6, 1),
        'last_reading': date(2013, 1, 31),
        'country': 'Switzerland'
    },
    'ECO_3': {
        'first_reading': date(2012, 7, 26),
        'last_reading': date(2013, 1, 31),
        'country': 'Switzerland'
    },
    'ECO_4': {
        'first_reading': date(2012, 7, 26),
        'last_reading': date(2013, 1, 31),
        'country': 'Switzerland'
    },
    'ECO_5': {
        'first_reading': date(2012, 7, 26),
        'last_reading': date(2013, 1, 31),
        'country': 'Switzerland'
    },
    'ECO_6': {
        'first_reading': date(2012, 7, 26),
        'last_reading': date(2013, 1, 31),
        'country': 'Switzerland'
    }
    }

    ECO_metadata = pd.DataFrame(houses).T
    ECO_metadata.reset_index(inplace=True)
    ECO_metadata.rename(columns={'index': 'name'}, inplace=True)

    return ECO_metadata


def LERTA_metadata():
   

    # read data
    lerta = pd.read_pickle(DATA_PATH+"LERTA.pkl")



    houses = {
        'LERTA_1': {
            'first_reading': pd.to_datetime(lerta["LERTA_1"]["AGGREGATE"].index).min().date(),
            'last_reading': pd.to_datetime(lerta["LERTA_1"]["AGGREGATE"].index).max().date(),
            'country': 'Poland',
            
        },
        'LERTA_2': {
            'first_reading': pd.to_datetime(lerta["LERTA_2"]["AGGREGATE"].index).min().date(),
            'last_reading': pd.to_datetime(lerta["LERTA_2"]["AGGREGATE"].index).max().date(),
            'country': 'Poland'
        },
        'LERTA_3': {
            'first_reading': pd.to_datetime(lerta["LERTA_3"]["AGGREGATE"].index).min().date(),
            'last_reading': pd.to_datetime(lerta["LERTA_3"]["AGGREGATE"].index).max().date(),
            'country': 'Poland'
        },
        'LERTA_4': {
            'first_reading': pd.to_datetime(lerta["LERTA_4"]["AGGREGATE"].index).min().date(),
            'last_reading': pd.to_datetime(lerta["LERTA_4"]["AGGREGATE"].index).max().date(),
            'country': 'Poland'
        },

    }


    LERTA_metadata = (pd.DataFrame(houses).T).reset_index()
    LERTA_metadata.rename(columns={'index': 'name'}, inplace=True)


    return LERTA_metadata

def UKDALE_metadata():
   
    with open(DATA_PATH+"UKDALE/metadata/dataset.yaml", 'r') as file:
        data = yaml.safe_load(file)

    # get lat and lon from yaml file
    lat  = data["geo_location"]["latitude"]
    lon = data["geo_location"]["longitude"]



    house_data = {}
    # go over all houses and get metadata
    for file in os.listdir(DATA_PATH+"UKDALE/metadata/"):
        if file.endswith(".yaml") and "building" in file:
            # print(file)
            with open(DATA_PATH+"UKDALE/metadata/" + file, 'r') as stream:
                try:
                    data = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            
            start = data["timeframe"]["start"].split("T")[0]
            end = data["timeframe"]["end"].split("T")[0]
            heating = np.nan
            occupants = np.nan
            if "heating" in data:
                heating = data["heating"][0]

            if "n_occupants" in data:
                occupants = data["n_occupants"]

            name = file.split(".")[0]
            name = "UKDALE_"+name[-1]
            house_data[name] = {
                "first_reading": start,
                "last_reading": end,
                "heating": heating,
                "occupancy": occupants,
                "lat": lat,
                "lon": lon,
                "country": "United Kingdom",
            }


    # convert to dataframe and reset index and rename columns
    UKDALE_metadata = pd.DataFrame(house_data).transpose()
    UKDALE_metadata.sort_index(inplace=True)
    UKDALE_metadata.reset_index(inplace=True)
    UKDALE_metadata.rename(columns={'index': 'name'}, inplace=True)

    return UKDALE_metadata

def generate_metadata(save=True):
    """Generate metadata for all datasets and save to parquet file if save is True"""
    # generate metadata for all datasets
    HUE_meta = HUE_metadata()
    REFIT_meta = REFIT_metadata()
    UCIML_meta = UCIML_metadata()
    HES_meta = HES_metadata()
    ECO_meta = ECO_metadata()
    LERTA_meta = LERTA_metadata()
    UKDALE_meta = UKDALE_metadata()

    # concat all metadata
    metadata = pd.concat([HUE_meta, REFIT_meta, UCIML_meta, HES_meta, ECO_meta, LERTA_meta, UKDALE_meta], ignore_index=True, axis=0)
    metadata.reset_index(inplace=True, drop=True)

    # convert first and last reading to datetime
    metadata["first_reading"] = pd.to_datetime(metadata["first_reading"])
    metadata["last_reading"] = pd.to_datetime(metadata["last_reading"])
  
    # save to parquet
    if save:
        metadata.to_parquet(SAVE_PATH + "residential_metadata.parquet")

    return metadata
                

if __name__ == "__main__":
    """Generate metadata for all datasets and save to parquet file if --save is passed as argument"""
    parser = argparse.ArgumentParser(description='Process data path and save path.')
    parser.add_argument('datapath', type=str, nargs='?', default='./energy-knowledge-graph/data/metadata/datasets/',
                        help='Path to the data')
    parser.add_argument('savepath', type=str, nargs='?', default='./energy-knowledge-graph/data/metadata/',
                        help='Path to save the results')
    parser.add_argument('--save', action='store_true', 
                        help='Save the result to parquet file if this argument is passed')
    args = parser.parse_args()

    DATA_PATH = args.datapath
    SAVE_PATH = args.savepath
    generate_metadata(save=args.save)

    # python generate_metadata.py path/to/data path/to/save --save

    # python generate_metadata.py path/to/data path/to/save

    # python generate_metadata.py --save to use default paths



    # generate_metadata()

    