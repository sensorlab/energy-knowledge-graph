import pandas as pd
from helper_functions import watts2kwh, save_to_pickle




# renames columns to appliance names and returns a list of dataframes where index is house number
def rename_columns(df: pd.DataFrame) -> list:

    house_dfs = []
    for house in df["house"].unique():
        house_df = df[df["house"] == house].copy()
        house_df.drop(columns=["house"], inplace=True)
        house_df /= 1000 # convert to kWh
        device_names = appliances[house-1].split(",")
        if device_names != None:
            house_df.columns = appliances[house-1].split(",")
            
        house_dfs.append(house_df)
    return house_dfs

# appliance names for each house
appliances = [
        'aggregate, fridge, chest freezer, upright freezer, tumble dryer, washing machine, dishwasher, computer site, television site, electric heater',
        'aggregate, fridge-freezer, washing machine, dishwasher, television, microwave, toaster, hi-fi, kettle, oven extractor fan',
        'aggregate, toaster, fridge-freezer, freezer, tumble dryer, dishwasher, washing machine, television, microwave, kettle',
        'aggregate, fridge, freezer, fridge-freezer, washing machine (1), washing machine (2), computer site, television site, microwave, kettle',
        'aggregate, fridge-freezer, tumble dryer 3, washing machine, dishwasher, computer site, television site, combination microwave, kettle, toaster',
        'aggregate, freezer (utility room), washing machine, dishwasher, mjy computer, television site, microwave, kettle, toaster, pgm computer',
        'aggregate, fridge, freezer (garage), freezer, tumble dryer, washing machine, dishwasher, television site, toaster, kettle',
        'aggregate, fridge, freezer, dryer, washing machine, toaster, computer, television site, microwave, kettle',
        'aggregate, fridge-freezer, washer dryer, washing machine, dishwasher, television site, microwave, kettle, hi-fi, electric heater',
        'aggregate, magimix (blender), freezer, chest freezer (in garage), fridge-freezer, washing machine, dishwasher, television site, microwave, kenwood kmix',
        'aggregate, fridge, fridge-freezer, washing machine, dishwasher, computer site, microwave, kettle, router, hi-fi',
        'aggregate, fridge-freezer, television site(lounge), microwave, kettle, toaster, television site (bedroom), not used, not used, not used',
        'aggregate, television site, unknown, washing machine, dishwasher, tumble dryer, television site, computer site, microwave, kettle',
        None,
        'aggregate, fridge-freezer, tumble dryer, washing machine, dishwasher, computer site, television site, microwave, kettle, toaster',
        'aggregate, fridge-freezer (1), fridge-freezer (2), electric heater (1)?, electric heater (2), washing machine, dishwasher, computer site, television site, dehumidifier/heater',
        'aggregate, freezer (garage), fridge-freezer, tumble dryer (garage), washing machine, computer site, television site, microwave, kettle, plug site (bedroom)',
        'aggregate, fridge(garage), freezer(garage), fridge-freezer, washer dryer(garage), washing machine, dishwasher, desktop computer, television site, microwave',
        'aggregate, fridge & freezer, washing machine, television site, microwave, kettle, toaster, bread-maker, lamp (80watts), hi-fi',
        'aggregate, fridge, freezer, tumble dryer, washing machine, dishwasher, computer site, television site, microwave, kettle',
        'aggregate, fridge-freezer, tumble dryer, washing machine, dishwasher, food mixer, television, kettle/toaster, vivarium, pond pump',
]

def parse_REFIT(data_path, save_path):
    # read data
    df = pd.read_parquet(data_path).set_index("timestamp")
    # remove metadata
    df.drop(columns=['occupancy', 'construction_year',
        'appliances_owned', 'house_type', 'house_size', 'country', 'location',
        'lat', 'lon', 'tz', 'appliances', 'is_holiday', 'weekday', 'is_weekend',
        'day_percent', 'week_percent', 'year_percent', 'solar_altitude',
        'solar_azimuth', 'solar_radiation'], inplace=True)

    houses_df = rename_columns(df)
   

    # dict to store dataframes for each house
    houses_data = {}


    # populate dict of dataframes for each house
    for i, house in enumerate(houses_df):
        if i >= 13:
            i=i+1
        name = "REFIT_" + str(i+1)
        data = {}
        #  add dataframes for each appliance
        for col in house.columns:
            data[col] = pd.DataFrame(house[col])

            houses_data[name] = data


    # save data
    save_to_pickle(houses_data, save_path)
