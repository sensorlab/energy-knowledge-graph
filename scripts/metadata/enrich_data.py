import pandas as pd
import requests
import numpy as np
import pycountry
from rasterio.warp import transform
import rasterio
from datetime import date
import holidays

# path to the folder where the data is stored
DATA_PATH = "./Energy_graph/data/metadata/"

# get temperature, relative humidity, precipitation, cloudcover(%)  from openmeteo api for given lat, lon and optional start and end date
def get_weather_data(lat, lon, start_date="2010-01-01", end_date="2023-01-01"):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relativehumidity_2m,precipitation,cloudcover&timezone=auto"
    try:
        response = requests.get(url)
        # Raise an exception if the response status is not 200 (HTTP_OK)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return None
    except requests.exceptions.RequestException as err:
        print(f'Error occurred: {err}')
        return None

    # No error was raised, so we can return the JSON data
    return response.json()


# get location data from openstreetmaps api for given lat, lon
def get_location_data(lat, lon):
    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&accept-language=en'
    try:
        res = requests.get(url)
        # Raise an exception if the response status is not 200 (HTTP_OK)
        res.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        return None
    except requests.exceptions.RequestException as err:
        print(f'Error occurred: {err}')
        return None

    # No error was raised, so we can return the JSON data
    return res.json()


# parse weather data into hourly and daily dataframes
def parse_weather_data(weather_data):
    df_hourly = pd.DataFrame(weather_data["hourly"])
    df_hourly["time"] = pd.to_datetime(df_hourly["time"])
    df_hourly.set_index("time", inplace=True)

    # resample to daily
    df_daily = df_hourly.resample('D').agg({'temperature_2m': np.mean, 'relativehumidity_2m': np.mean, 'precipitation': np.sum, 'cloudcover': np.mean})

    return df_hourly, df_daily
# calculate average weather values for each day of the year for the past 13 years
def average_day_of_the_year(df_daily):
    yearly = df_daily.groupby(df_daily.index.dayofyear).mean()
    return yearly
# calculate average weather values for each day(hourly) of the month for all months
def average_day_of_the_month(df_hourly):
    df_avg_day_hourly = df_hourly.groupby([df_hourly.index.month, df_hourly.index.hour]).mean()
    df_avg_day_hourly.index.names = ['month', 'hour']

    return df_avg_day_hourly

# get GDP(PPP) for given country and year, the GDP is in 2017 USD 
def get_country_GDP(country : str, year: int) -> float:
    gpd_PPP =  pd.read_csv(DATA_PATH +"wages/gdp-per-capita-worldbank.csv")
    df = gpd_PPP[gpd_PPP["Year"] == year].drop(columns=["Year", "Entity"])

    
    df.reset_index(inplace=True, drop=True)
    df.set_index("Code", inplace=True)

    # get ISO-3166 alpha-3 country code from country name
    country_code = get_country_code(country)

    return df.loc[country_code].values[0]

# get average wages for given country and year, the data is only available for OECD countries
def get_average_wages(country: str, year: int) -> float:
    wages = pd.read_csv(DATA_PATH + "Wages/average_wages_OECD.csv")
    if year not in wages["TIME"].unique():
        raise ValueError("Year must be between 1991 and 2021")
    
    # drop unnecessary columns and set index to country name
    wages_processed = wages[wages["TIME"] == year].drop(columns=["TIME", "INDICATOR", "SUBJECT", "MEASURE", "FREQUENCY", "Flag Codes"]).set_index("LOCATION")
    
    # get ISO-3166 alpha-3 country code from country name
    country_code =get_country_code(country)
    return wages_processed.loc[country_code].values[0]


# get population density for given latitude and longitude
def get_population_density(latitude : float, longitude: float) -> float:
        
    # Open the dataset
    with rasterio.open(DATA_PATH+ "Population/Density_5min/gpw_v4_population_density_rev11_2020_2pt5_min.asc") as ds:

        # Convert your latitude and longitude to dataset's coordinate system
        x, y = transform('EPSG:4326', ds.crs, [longitude], [latitude])

        # Fetch the population density value at the given coordinates
        row, col = ds.index(x[0], y[0])
        population_density = ds.read(1)[row, col]

    # return population density in people per square kilometer
    return population_density

# get elevation for given latitude and longitude
def get_elevation(lat : float, lon: float) -> float:
    # get elevation from open-meteo api
    elevation = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}")

    # return elevation in meters
    return elevation.json()["elevation"][0]


def get_country_code(country):
    return pycountry.countries.search_fuzzy(country)[0].alpha_3


# get education level for given country and year
def get_education_level(country: str, year: int) -> float:
    df = pd.read_csv(DATA_PATH + "Population/Education/adult_education_levels.csv")
    if year not in df["TIME"].unique():
        raise ValueError("Year must be between 2018 and 2021")
    
    # drop unnecessary columns and set index to country code
    df = df[df["TIME"] == year].drop(columns=["TIME", "Flag Codes", "MEASURE", "FREQUENCY", "INDICATOR"]).set_index("LOCATION")

    # get ISO-3166 alpha-3 country code from country name
    country_code = get_country_code(country)

    # ages 25-64
    # % of population with below upper secondary education, % of population with upper secondary, % of population with tertiary education
    return df.loc[country_code].values[0][1],df.loc[country_code].values[2][1],  df.loc[country_code].values[1][1]
    
# returns the semester for given date for gas and electricity prices
def get_semester(date: date):
    if date.month < 7:
        return str(date.year) + "-S1"
    else:
        return str(date.year) + "-S2"

# get electricity price for given country and date the price is in PPS per kWh and data is biannual
def get_electricity_price(country: str, date: date) -> float:
    if date.year < 2007 or date.year > 2022:
        raise ValueError("Year must be between 2007 and 2022")
    
    df = pd.read_csv(DATA_PATH + "Energy/Electricity_prices_EEA.csv")
    df.drop(columns=["LAST UPDATE", "freq", "product","OBS_FLAG", "DATAFLOW", "currency", "tax", "unit"])
    # average electricity prices in PPS per kWh for diffrent yearly consumption levels
    df_grouped = df.groupby(['geo', 'TIME_PERIOD'])['OBS_VALUE'].mean()
    # get ISO-3166 alpha-2 country code from country name
    country_code = pycountry.countries.search_fuzzy(country)[0].alpha_2
    semester = get_semester(date)
    
    return df_grouped.loc[country_code, semester]

# get gas price for given country and date the price is in PPS per kWh and data is biannual
def get_gas_price(country: str, date: date) -> float:
    if date.year < 2007 or date.year > 2022:
        raise ValueError("Year must be between 2007 and 2022")
    
    df = pd.read_csv(DATA_PATH + "Energy/Gas_prices_EEA.csv")

    df= df.drop(columns=["DATAFLOW", "LAST UPDATE", "freq", "product", "nrg_cons", "unit", "tax", "currency", "OBS_FLAG"])
    df = df.set_index(["geo", "TIME_PERIOD"])

    country_code = pycountry.countries.search_fuzzy(country)[0].alpha_2
    semester = get_semester(date)
    # TODO handle missing values
    return df.loc[country_code, semester].values[0]

# Explanation:
# Heating Degree Days (HDD) index:  the severity of the cold in a specific time period taking into consideration outdoor temperature and average room temperature (in other words the need for heating).
# The calculation of HDD relies on the base temperature, defined as the lowest daily mean air temperature not leading to indoor heating.
# The value of the base temperature depends in principle on several factors associated with the building and the surrounding environment.
#  By using a general climatological approach, the base temperature is set to a constant value of 15°C in the HDD calculation. 
# If Tm ≤ 15°C Then [HDD = ∑i(18°C - Tim)] Else [HDD = 0] where Tim is the mean air temperature of day i.

# Examples: If the daily mean air temperature is 12°C, for that day the value of the HDD index is 6 (18°C-12°C). If the daily mean air temperature is 16°C, for that day the HDD index is 0.

 
# Cooling degree days (CDD) index:  the severity of the heat in a specific time period taking into consideration outdoor temperature and average room temperature (in other words the need for cooling).
# The calculation of CDD relies on the base temperature, defined as the highest daily mean air temperature not leading to indoor cooling.
# The value of the base temperature depends in principle on several factors associated with the building and the surrounding environment.
# By using a general climatological approach, the base temperature is set to a constant value of 24°C in the CDD calculation.

# If Tm ≥ 24°C Then [CDD = ∑iTim - 21°C)] Else [CDD = 0] where Tim is the mean air temperature of day i.

# Examples: If the daily mean air temperature is 26°C, for that day the value of the CDD index is 5 (26°C-21°C).
#  If the daily mean air temperature is 22°C, for that day the CDD index is 0.

# https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm
def get_cooling_and_heating_degree_days(country : str, year: int) -> float:
    df = pd.read_csv(DATA_PATH + "Energy/Heating_cooling_index.csv")

    # drop useless columns
    df.drop(columns=["DATAFLOW", "LAST UPDATE","unit", "freq", "OBS_FLAG"], inplace=True)
    df_pivot = df.pivot(index=['geo', 'TIME_PERIOD'], columns='indic_nrg', values='OBS_VALUE')
    
    country_code = pycountry.countries.search_fuzzy(country)[0].alpha_2
    

    return df_pivot.loc[country_code, year]["CDD"], df_pivot.loc[country_code, year]["HDD"]

# return public holidays for given country
def get_public_holidays(country : str) -> int:
    country_code = pycountry.countries.search_fuzzy(country)[0].alpha_2
    return holidays.country_holidays(country_code)


if __name__ == "__main__":
    # TODO create table for PostgreSQL
    