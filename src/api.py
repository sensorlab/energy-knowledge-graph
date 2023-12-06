from sqlalchemy import text
from sqlalchemy.engine import Connection
import json
from math import isnan
from .enrich_data import create_location_dict, create_weather_dict
import re

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


create_households_table_sql = text('''
CREATE TABLE "households" (
	"household_id" bigserial NOT NULL,
	"location_id" bigserial NOT NULL,
	"name" varchar NOT NULL UNIQUE,
	"house_type" varchar,
	"first_reading" DATE NOT NULL,
	"last_reading" DATE NOT NULL,
	"occupancy" int,
	"facing" varchar,
	"rental_units" int,
	"evs" int,
    "consumption" float,                                   
	CONSTRAINT "households_pk" PRIMARY KEY ("household_id")
) WITH (
  OIDS=FALSE
);
''')

create_locations_table_sql = text('''
CREATE TABLE "locations" (
	"location_id" bigserial NOT NULL,
	"weather_id" bigserial NOT NULL,
	"continent" varchar,
	"country" varchar,
	"country_code" varchar,
	"region" varchar,
	"city" varchar,
	"street" varchar,
	"timezone" varchar,
	"latitude" float,
	"longitude" float,
	"gdp" float,
	"wages" float,
	"population_density" float,
	"elevation" float,
	"education_level_low" float,
    "education_level_medium" float,
    "education_level_high" float,
	"electricity_price" float,
	"gas_price" float,
	"cdd" float,
	"hdd" float,
	"holidays" DATE[],
    "carbon_intesity" float,
	CONSTRAINT "locations_pk" PRIMARY KEY ("location_id")
) WITH (
  OIDS=FALSE
);
''')

create_devices_table_sql = text('''
CREATE TABLE "devices" (
	"device_id" bigserial NOT NULL,
	"household_id" bigserial NOT NULL,
	"name" varchar NOT NULL,
	"loadprofile_daily" float[] NOT NULL,
	"loadprofile_weekly" float[] NOT NULL,
	"loadprofile_monthly" float[] NOT NULL,
    "daily_consumption" float,
    "event_consumption" float,                                
	CONSTRAINT "device_pk" PRIMARY KEY ("device_id")
) WITH (
  OIDS=FALSE
);
''')
create_weather_table_sql = text('''
CREATE TABLE "weather" (
	"weather_id" bigserial,
	"yearly" float[],
	"hourly" float[],
	CONSTRAINT "weather_pk" PRIMARY KEY ("weather_id")
) WITH (
  OIDS=FALSE
);
''')

alter_tables = text('''                 
ALTER TABLE "households" ADD CONSTRAINT "households_fk0" FOREIGN KEY ("location_id") REFERENCES "locations"("location_id");

ALTER TABLE "devices" ADD CONSTRAINT "device_fk0" FOREIGN KEY ("household_id") REFERENCES "households"("household_id");

ALTER TABLE "locations" ADD CONSTRAINT "locations_fk0" FOREIGN KEY ("weather_id") REFERENCES "weather"("weather_id");
  ''')



def ensure_tables(conn: Connection) -> None:
    conn.execute(create_locations_table_sql)
    conn.execute(create_households_table_sql)
    conn.execute(create_devices_table_sql)
    conn.execute(create_weather_table_sql)
    conn.execute(alter_tables)


#significant_fields = ('country', 'state', 'municipality', 'city', 'town', 'village', 'suburb', 'neighbourhood')

def get_or_create_location_id(conn: Connection, household:dict) -> int:
    """We will try to find area that exactly matches"""

    assert household.get('country', None), 'Country is mandatory'


    query_location_sql = text('''
        SELECT location_id FROM locations WHERE country = :country AND latitude = :latitude AND longitude = :longitude
    ''')

    insert_location_sql = text('''
        INSERT INTO locations (weather_id, continent, country, country_code, region, city, street, timezone, latitude, longitude, gdp, wages, population_density, elevation, education_level_low, education_level_medium, education_level_high, electricity_price, gas_price, cdd, hdd, holidays, carbon_intesity)
        VALUES (:weather_id, :continent, :country, :country_code, :region, :city, :street, :timezone, :latitude, :longitude, :gdp, :wages, :population_density, :elevation, :education_level_low, :education_level_medium, :education_level_high, :electricity_price, :gas_price, :cdd, :hdd, :holidays, :carbon_intesity)
        RETURNING location_id;
    ''')

    insert_weather_sql = text('''
        INSERT INTO weather (yearly, hourly)
        VALUES (:yearly, :hourly)
        RETURNING weather_id;
    ''')
    # queary location data to check if location already exists in database
    locations = conn.execute(query_location_sql, dict(country=household["country"], latitude=household["lat"], longitude=household["lon"])).scalars().all()

    # Sanity checks
    assert len(locations) < 2, f'There are two locations with identical location: {locations}'

    # Does entry already exist? If it does, use it, otherwise create new one.
    if len(locations) > 0:
        location_id = locations[0]
        return location_id
    
    # get location data
    lat, lon, country = household['lat'], household['lon'], household['country']
    city = household.get('city', None)
    if lat is None or lon is None or isnan(lat) or isnan(lon):
        location_data = create_location_dict(country, household["first_reading"], 0)
    else:
        location_data = create_location_dict(country, household["first_reading"], 1, lat, lon)
    # special case if we have city info from somwhere else but we dont have accurate coordinates
    if location_data["city"] is None:
        location_data["city"] = city
    
    # cast average wages to int
    if location_data["wages"] is not None:
        location_data["wages"] = int(location_data["wages"])

    # get weather data TODO for now just fill with null
    weather_id = conn.execute(insert_weather_sql, dict(yearly=None, hourly=None)).scalar_one()
    # Create entry in DB
    location_id = conn.execute(insert_location_sql,
                                dict(
        weather_id=weather_id,
        continent=location_data["continent"],
        country=location_data["country"], 
        country_code=location_data["country_code"], 
        region=location_data["region"], 
        city=location_data["city"], 
        street=location_data["street"], 
        timezone=location_data["timezone"], 
        latitude=location_data["latitude"], 
        longitude=location_data["longitude"], 
        gdp=location_data["GDP"], 
        wages=location_data["wages"], 
        population_density=location_data["population_density"], 
        elevation=location_data["elevation"], 
        education_level_low=location_data["education_level"][0], 
        education_level_medium=location_data["education_level"][0],
        education_level_high=location_data["education_level"][0],
        electricity_price=location_data["electricity_price"], 
        gas_price=location_data["gas_price"], 
        cdd=location_data["CDD"], 
        hdd=location_data["HDD"], 
        holidays=location_data["public_holidays"],
        carbon_intesity=location_data["carbon_intesity"])
                                ).scalar_one()


    return location_id


def query_osm_metadata(lat:float, lon:float) -> dict:
    """OSM - OpenStreetMap"""
    import requests
    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&accept-language=en'
    res = requests.get(url)
    return res.json()


def get_or_create_household_id(conn: Connection, household: dict, consumption : float) -> int:
    """Currently there is no way to validate duplication of the entries."""

    query_household_sql = text('''
        SELECT household_id FROM households WHERE name = :name
    ''')

    # Does entry already exist? If it does, use it, otherwise create new one.
    households = conn.execute(query_household_sql, dict(name=household["name"])).scalars().all()

    if len(households) > 0:
        household_id = households[0]
        return household_id

    insert_household_sql = text('''
        INSERT INTO households (location_id, name, house_type, first_reading, last_reading, occupancy, facing, rental_units, evs, consumption)
        VALUES (:location_id, :name, :house_type, :first_reading, :last_reading, :occupancy, :facing, :rental_units, :evs, :consumption)
        RETURNING household_id;
    ''')



    
    # Obtain location ID
    location_id = get_or_create_location_id(conn, household)

    # Convert occupancy to int if it is not NaN
    if not isnan(household["occupancy"]):
        household["occupancy"] = int(household["occupancy"])
    else:
        household["occupancy"] = None

    # Convert rental_units to int if it is not NaN
    if not isnan(household["rental_units"]):
        household["rental_units"] = int(household["rental_units"])
    else:
        household["rental_units"] = None

    # Convert evs to int if it is not NaN
    if not isnan(household["EVs"]):
        household["EVs"] = int(household["EVs"])
    else:
        household["EVs"] = None

    # Convert consumption to float if it is not None
    if consumption is not None:
        consumption = float(consumption)


    # Create entry in DB
    household_id = conn.execute(insert_household_sql, dict(
        location_id=location_id,
        name=household["name"],
        house_type=household["house_type"],
        first_reading=household["first_reading"],
        last_reading=household["last_reading"],
        occupancy=household["occupancy"],
        facing=household["facing"],
        rental_units=household["rental_units"],
        evs=household["EVs"],
        consumption=consumption
        )).scalar_one()
    
    
    return household_id



def get_or_create_device_id(conn:Connection, device:str, household_id:int, data:dict, daily_consumption:float, event_consumption:float) -> int:
    insert_device_sql = text('''
        INSERT INTO devices (household_id, name, loadprofile_daily, loadprofile_weekly, loadprofile_monthly, daily_consumption, event_consumption)
        VALUES (:household_id, :name, :loadprofile_daily, :loadprofile_weekly, :loadprofile_monthly, :daily_consumption, :event_consumption)
        RETURNING device_id;
    ''')

    query_device_sql = text('''
        SELECT device_id FROM devices WHERE household_id = :household_id AND name = :name
    ''')

    if daily_consumption is not None:
        daily_consumption = float(daily_consumption)
    if event_consumption is not None:
        event_consumption = float(event_consumption)

    # Does entry already exist? If it does, use it, otherwise create new one.
    devices = conn.execute(query_device_sql, dict(household_id=household_id, name=device)).scalars().all()
    if len(devices) > 0:
        device_id = devices[0]
        return device_id
    # Create entry in DB
    device_id = conn.execute(insert_device_sql, dict(
        household_id=household_id, 
        name = preprocess_string(device),
        # loadprofile_daily = list(data[device]["daily"].flatten()),
        # loadprofile_weekly = list(data[device]["weekly"].flatten()),
        # loadprofile_monthly = list(data[device]["monthly"].flatten()),
        loadprofile_daily = data[device]["daily"].flatten().astype(float).tolist(),
        loadprofile_weekly = data[device]["weekly"].flatten().astype(float).tolist(),
        loadprofile_monthly = data[device]["monthly"].flatten().astype(float).tolist(),
        daily_consumption = daily_consumption,
        event_consumption = event_consumption

        
        )).scalar_one()
    return device_id

def get_or_create_weather_id(conn:Connection, weather:dict) -> int:
    # TODO: finish this


    insert_weather_sql = text('''
        INSERT INTO weather (meta)
        VALUES (:meta)
        RETURNING id;
    ''')

    # Create entry in DB
    weather_id = conn.execute(insert_weather_sql, dict(meta=weather)).scalar_one()
    return weather_id