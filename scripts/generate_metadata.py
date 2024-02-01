import pandas as pd
import os
import numpy as np
from datetime import date
import yaml
import sys
import argparse
from io import StringIO
from pathlib import Path

DATA_PATH : Path = Path("./energy-knowledge-graph/data/metadata/datasets/").resolve()
SAVE_PATH : Path = Path("./energy-knowledge-graph/data/metadata/").resolve()

def HUE_metadata():
    # read data
    HUE_metadata = pd.read_parquet(DATA_PATH / "HUE_metadata.parquet")

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
    REFIT_metadata = pd.read_parquet(DATA_PATH / "refit_metadata.parquet")

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
    data = pd.read_pickle(DATA_PATH / "REFIT.pkl")
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
    data_uciml = pd.read_parquet(DATA_PATH / "uciml_household.parquet")
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
        "city": "Paris",
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
    lerta = pd.read_pickle(DATA_PATH / "LERTA.pkl")



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
   
    with open(DATA_PATH / "UKDALE/metadata/dataset.yaml", 'r') as file:
        data = yaml.safe_load(file)

    # get lat and lon from yaml file
    lat  = data["geo_location"]["latitude"]
    lon = data["geo_location"]["longitude"]



    house_data = {}
    # go over all houses and get metadata
    for file in os.listdir(DATA_PATH / "UKDALE/metadata/"):
        if file.endswith(".yaml") and "building" in file:
            # print(file)
            with open(DATA_PATH / "UKDALE/metadata" / file, 'r') as stream:
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
            # skip due to lacking device submeter data(devices grouped together)
            if name == "UKDALE_4":
                continue
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

def DRED_metadata():
    dred ={
    "name" : "DRED_1",
    "first_reading" : date(2015, 7, 5),
    "last_reading" : date(2015, 12, 5),
    "country" : "Netherlands",
    }
    dred = pd.DataFrame(dred, index=[0])
    return dred

def REDD_metadata():
    redd_data = pd.read_pickle(DATA_PATH / "REDD.pkl")

    redd = {}

    for name, value in redd_data.items():
        redd[name] =  {
            "first_reading": value["aggregate"].index.date.min(),
            "last_reading": value["aggregate"].index.date.max(),
            "lat" : 42.360338,
            "lon" : -71.064709,
            "country" : "United States",
        }
        
        
    redd = pd.DataFrame(redd).T
    redd.index.name = "name"
    redd.reset_index(inplace=True)
    return redd

def IAWE_metadata():
    iawe = {
    "name": "IAWE_1",
    "country": "India",
    "lat": 28.644800,
    "lon": 77.216721,
    "first_reading": date(2013,7,13),
    "last_reading": date(2013, 8,4),
    }

    df = pd.DataFrame(iawe, index=[0])
    return df

def DEKN_metadata():
    dekn = pd.read_pickle(DATA_PATH / "DEKN.pkl")
    data = {}
    for house in dekn:
        data[house] = {
            "name": house,
            "first_reading": pd.to_datetime(dekn[house]["aggregate"].index.date.min()),
            "last_reading": pd.to_datetime(dekn[house]["aggregate"].index.date.max()),
            "country": "Germany",
            "lat" : 47.66033,
            "lon" : 9.17582,
        }
    dekn = pd.DataFrame(data).T.reset_index(drop=True)

    return dekn

def HEART_metadata():
    data = {}

    data = {
        "HEART_7"  : {
            "name" : "HEART_7",
            "first_reading" : date(2022,7,7),
            "last_reading"  : date(2022,8,8),
            "country"       : "Greece",
            
        },
        
        "HEART_33"  : {
            "name": "HEART_33",
            "first_reading" : date(2022,7,7),
            "last_reading"  : date(2022,8,8),
            "country"       : "Greece",
            
        }

    }

    heart = pd.DataFrame.from_dict(data).T
    heart.reset_index(drop=True, inplace=True)
    return heart

def SUST1_metadata():
    # drop unnecessary columns TODO UPDATE PATH
    df = pd.read_csv(DATA_PATH / "demographics_SUST1.csv", delimiter=";").drop(columns=["Unnamed: 0", "# Adults", "# Children", "Rented?","Start Feedback", "End Feedback", "Contracted Power (kVA)"])
    # rename columns to match the other metadata
    df.rename(columns={"# People": "occupancy", "Type (A/H)": "house_type", "Start Measuring" : "first_reading", "End Measuring" : "last_reading", "SustData IID": "name"},inplace=True)
    # convert to datetime
    df["first_reading"] = pd.to_datetime(df["first_reading"])
    df["last_reading"] = pd.to_datetime(df["last_reading"])
    # convert to match the other metadata
    df["house_type"] = df["house_type"].apply(lambda x: "apartment" if x == "A" else "house")
    # convert to match the other metadata
    df["name"] = "SUST1_"+df["name"].astype(str)
    # add country and location
    df["country"] = "Portugal"
    df["lat"] = 32.66
    df["lon"] = -16.917012
    # drop the 4 rows with missing data
    df.drop([50,51,52,53], inplace=True)

    return df

def SUST2_metadata():
    data = {
        "name" : "SUST2_1",
        "first_reading" : date(2016, 10, 6),
        "last_reading" : date(2016, 12, 31),
        "country" : "Portugal",
        "occupancy" : 3,
        "house_type" : "house",
    }

def DEDDIAG_metadata():
    data = {
    "name" : "DEDDIAG_8",
    "first_reading" : date(2017, 9, 12),
    "last_reading" : date(2018, 7, 28),
    "country" : "Germany",
    }   

    return pd.DataFrame(data, index=[0])

def ENERTALK_metadata():
    
    html_string = """
    <table class="data last-table"><thead class="c-article-table-head"><tr><th class="u-text-left "><p>House code</p></th><th class="u-text-left "><p>Start date</p></th><th class="u-text-left "><p>End date</p></th><th class="u-text-left "><p>Duration (days)</p></th><th class="u-text-left "><p>Refrigerator</p></th><th class="u-text-left "><p>Kimchi refrigerator</p></th><th class="u-text-left "><p>Rice cooker</p></th><th class="u-text-left "><p>Washing machine</p></th><th class="u-text-left "><p>TV</p></th><th class="u-text-left "><p>Microwave</p></th><th class="u-text-left "><p>Water-purifier</p></th></tr></thead><tbody><tr><td class="u-text-left "><p>00</p></td><td class="u-text-left "><p>2016-11-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>91</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td></tr><tr><td class="u-text-left "><p>01</p></td><td class="u-text-left "><p>2016-10-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>122</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>02</p></td><td class="u-text-left "><p>2016-10-01</p></td><td class="u-text-left "><p>2016-10-31</p></td><td class="u-text-left "><p>30</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>03</p></td><td class="u-text-left "><p>2016-10-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>122</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>04</p></td><td class="u-text-left "><p>2016-09-01</p></td><td class="u-text-left "><p>2016-11-30</p></td><td class="u-text-left "><p>90</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>05</p></td><td class="u-text-left "><p>2016-09-03</p></td><td class="u-text-left "><p>2016-10-31</p></td><td class="u-text-left "><p>58</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>06</p></td><td class="u-text-left "><p>2016-09-01</p></td><td class="u-text-left "><p>2016-10-15</p></td><td class="u-text-left "><p>44</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td></tr><tr><td class="u-text-left "><p>07</p></td><td class="u-text-left "><p>2016-12-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>61</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>08</p></td><td class="u-text-left "><p>2016-12-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>61</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>09</p></td><td class="u-text-left "><p>2016-10-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>122</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>10</p></td><td class="u-text-left "><p>2016-10-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>122</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>11</p></td><td class="u-text-left "><p>2017-04-01</p></td><td class="u-text-left "><p>2017-04-30</p></td><td class="u-text-left "><p>29</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>12</p></td><td class="u-text-left "><p>2016-10-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>122</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>13</p></td><td class="u-text-left "><p>2016-11-02</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>90</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>14</p></td><td class="u-text-left "><p>2016-10-01</p></td><td class="u-text-left "><p>2017-01-20</p></td><td class="u-text-left "><p>111</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>15</p></td><td class="u-text-left "><p>2017-03-15</p></td><td class="u-text-left "><p>2017-04-30</p></td><td class="u-text-left "><p>46</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>16</p></td><td class="u-text-left "><p>2016-09-01</p></td><td class="u-text-left "><p>2016-11-15</p></td><td class="u-text-left "><p>75</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>17</p></td><td class="u-text-left "><p>2016-11-03</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>89</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>18</p></td><td class="u-text-left "><p>2016-09-01</p></td><td class="u-text-left "><p>2016-10-19</p></td><td class="u-text-left "><p>48</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>19</p></td><td class="u-text-left "><p>2016-09-01</p></td><td class="u-text-left "><p>2016-10-31</p></td><td class="u-text-left "><p>60</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>20</p></td><td class="u-text-left "><p>2017-03-01</p></td><td class="u-text-left "><p>2017-04-30</p></td><td class="u-text-left "><p>60</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr><tr><td class="u-text-left "><p>21</p></td><td class="u-text-left "><p>2016-12-01</p></td><td class="u-text-left "><p>2017-01-31</p></td><td class="u-text-left "><p>61</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>O</p></td><td class="u-text-left "><p>X</p></td><td class="u-text-left "><p>X</p></td></tr></tbody></table>
    """

    # Read the HTML into a list of DataFrames
    df = pd.read_html(StringIO(html_string))[0]
    df.drop(columns=["Duration (days)","Refrigerator","Kimchi refrigerator","Rice cooker","Washing machine","TV","Microwave","Water-purifier"], inplace=True)
    df.rename(columns={"House code":"name", "Start date" : "first_reading", "End date": "last_reading"},inplace=True)

    # Convert the date columns to datetime
    df["first_reading"] = pd.to_datetime(df["first_reading"])
    df["last_reading"] = pd.to_datetime(df["last_reading"])

    # changed name so its the same as the other datasets
    df["name"] = "ENERTALK_" + df["name"].astype(str)
    df["country"] = "South Korea"
    return df

def ECDUY_metadata():
    data = pd.read_pickle(DATA_PATH / "ECDUY_metadata.pkl")
    df = pd.DataFrame(data).T.reset_index(drop=True)

    df["country"] = "Uruguay"
    df["city"] = "Montevideo"
    df["lat"] = -34.901112
    df["lon"] = -56.164532
    return df



def IDEAL_metadata():

    df = pd.read_csv(DATA_PATH / "IDEAL_metadata.csv")
    df["name"] = "IDEAL_" + df["homeid"].astype(str)

    # get coordinates for each location
    coordinates = {
        "Edinburgh" : (55.9533, -3.1883),
        "Midlothian" : (55.889829774, -3.067833062),
        "WestLothian": (55.916663, -3.499998),
        "EastLothian" : (55.916663, -2.749997),
        "Fife" : (56.249999, -3.1999992),
    }
    #  add coordinates and country data
    df["lat"] = df["location"].apply(lambda x: coordinates[x][0])
    df["lon"] = df["location"].apply(lambda x: coordinates[x][1])
    df["country"] = "United Kingdom"
    # rename columns to match other datasets and drop unnecessary columns
    df.rename(columns={"residents": "occupancy", "starttime": "first_reading","endtime" : "last_reading", "build_era": "construction_year", "hometype" : "house_type" }, inplace=True)
    df.drop(columns=["homeid",'install_type', "starttime_enhanced", "cohortid", "income_band","study_class","new_build_year", "smart_monitors","smart_automation","occupied_days","occupied_nights","outdoor_space","outdoor_drying", "urban_rural_class", "equivalised_income", "entry_floor", "urban_rural_name", "location", "occupancy"], inplace=True)
    # convert first and last reading to datetime
    df["first_reading"] = pd.to_datetime(df["first_reading"])
    df["last_reading"] = pd.to_datetime(df["last_reading"])
    # change house type to match other datasets
    df['house_type'] = df['house_type'].replace({
        'flat': 'apartment',
        'house_or_bungalow': 'house'
    })

    return df

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
    DRED_meta = DRED_metadata()
    REDD_meta = REDD_metadata()
    IAWE_meta = IAWE_metadata()
    DEKN_meta = DEKN_metadata()
    HEART_meta = HEART_metadata()
    SUST_meta = pd.concat([SUST1_metadata(), SUST2_metadata()], ignore_index=True, axis=0)
    DEDDIAG_meta = DEDDIAG_metadata()
    ENERTALK_meta = ENERTALK_metadata()
    ECDUY_meta = ECDUY_metadata()
    IDEAL_meta = IDEAL_metadata()

    # concat all metadata
    metadata = pd.concat(
        [
        HUE_meta,
        REFIT_meta,
        UCIML_meta,
        HES_meta,
        ECO_meta,
        LERTA_meta,
        UKDALE_meta,
        DRED_meta,
        REDD_meta,
        IAWE_meta,
        DEKN_meta,
        HEART_meta,
        SUST_meta,
        DEDDIAG_meta,
        ENERTALK_meta,
        ECDUY_meta,
        IDEAL_meta
        ],
         ignore_index=True, axis=0)
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

    DATA_PATH : Path = Path(args.datapath).resolve()
    SAVE_PATH : Path = Path(args.savepath).resolve()
    generate_metadata(save=args.save)

    # python generate_metadata.py path/to/data path/to/save --save

    # python generate_metadata.py path/to/data path/to/save

    # python generate_metadata.py --save to use default paths



    # generate_metadata()

    