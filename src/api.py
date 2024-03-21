from math import isnan

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .helper import preprocess_string
from .enrich_data import create_location_dict

# SQL queries
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
    "labeled" boolean,                        
    "house_size" float,           
	CONSTRAINT "households_pk" PRIMARY KEY ("household_id")
) WITH (
  OIDS=FALSE
);
''')

create_locations_table_sql = text('''
CREATE TABLE "locations" (
	"location_id" bigserial NOT NULL,
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
    "education_category" varchar,                                
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

alter_tables = text('''                 
ALTER TABLE "households" ADD CONSTRAINT "households_fk0" FOREIGN KEY ("location_id") REFERENCES "locations"("location_id");

ALTER TABLE "devices" ADD CONSTRAINT "device_fk0" FOREIGN KEY ("household_id") REFERENCES "households"("household_id");

  ''')


def ensure_tables(conn: Connection) -> None:
    """Ensure that the tables are created in the database"""
    conn.execute(create_locations_table_sql)
    conn.execute(create_households_table_sql)
    conn.execute(create_devices_table_sql)
    conn.execute(alter_tables)


# significant_fields = ('country', 'state', 'municipality', 'city', 'town', 'village', 'suburb', 'neighbourhood')

def get_or_create_location_id(conn: Connection, household: dict) -> int:
    """
    Get or create location ID if it doesn't exist yet
    ## Parameters
    `conn` : The connection to the database
    `household` : The household dictionary
    ## Returns
    `int` : The location ID

    """

    assert household.get('country', None), 'Country is mandatory'

    query_location_sql = text('''
        SELECT location_id FROM locations WHERE country = :country AND latitude = :latitude AND longitude = :longitude
    ''')

    insert_location_sql = text('''
        INSERT INTO locations (continent, country, country_code, region, city, street, timezone, latitude, longitude, gdp, wages, population_density, elevation, education_level_low, education_level_medium, education_level_high, education_category, electricity_price, gas_price, cdd, hdd, holidays, carbon_intesity)
        VALUES (:continent, :country, :country_code, :region, :city, :street, :timezone, :latitude, :longitude, :gdp, :wages, :population_density, :elevation, :education_level_low, :education_level_medium, :education_level_high, :education_category, :electricity_price, :gas_price, :cdd, :hdd, :holidays, :carbon_intesity)
        RETURNING location_id;
    ''')

    # queary location data to check if location already exists in database
    locations = conn.execute(query_location_sql, dict(country=household["country"], latitude=household["lat"],
                                                      longitude=household["lon"])).scalars().all()

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
    # special case if we have city info from somwhere else, but we don't have accurate coordinates
    if location_data["city"] is None:
        location_data["city"] = city

    # cast average wages to int
    if location_data["wages"] is not None:
        location_data["wages"] = int(location_data["wages"])

    # Create entry in DB
    location_id = conn.execute(insert_location_sql,
                               dict(
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
                                   education_level_medium=location_data["education_level"][1],
                                   education_level_high=location_data["education_level"][2],
                                   education_category=location_data["education_level"][3],
                                   electricity_price=location_data["electricity_price"],
                                   gas_price=location_data["gas_price"],
                                   cdd=location_data["CDD"],
                                   hdd=location_data["HDD"],
                                   holidays=location_data["public_holidays"],
                                   carbon_intesity=location_data["carbon_intesity"])
                               ).scalar_one()

    return location_id


def query_osm_metadata(lat: float, lon: float) -> dict:
    """
    OSM - OpenStreetMap
    Query OSM for metadata
    ## Parameters
    `lat` : Latitude
    `lon` : Longitude
    ## Returns
    `dict` : The metadata in JSON format as a dictionary
    """

    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&accept-language=en'
    res = requests.get(url)
    return res.json()


def get_or_create_household_id(conn: Connection, household: dict, consumption: float, labeled: bool) -> int:
    """
    Get or create household ID if it doesn't exist yet
    ## Parameters
    `conn` : The connection to the database
    `household` : The household dictionary
    `consumption` : The consumption of the household
    `labeled` : Whether the data is labeled
    ## Returns
    `int` : The household ID

    """

    query_household_sql = text('''
        SELECT household_id FROM households WHERE name = :name
    ''')

    # Does entry already exist? If it does, use it, otherwise create new one.
    households = conn.execute(query_household_sql, dict(name=household["name"])).scalars().all()

    if len(households) > 0:
        household_id = households[0]
        return household_id

    insert_household_sql = text('''
        INSERT INTO households (location_id, name, house_type, first_reading, last_reading, occupancy, facing, rental_units, evs, consumption, house_size, labeled)
        VALUES (:location_id, :name, :house_type, :first_reading, :last_reading, :occupancy, :facing, :rental_units, :evs, :consumption, :house_size, :labeled)
        RETURNING household_id;
    ''')

    # Obtain location ID
    location_id = get_or_create_location_id(conn, household)
    try:
        household["house_size"] = float(household["house_size"]) if household["house_size"] is not None else None
    except ValueError:
        household["house_size"] = None
    if household["house_size"] is not None and not isnan(household["house_size"]):
        household["house_size"] = float(household["house_size"])
    else:
        household["house_size"] = None

    # Convert occupancy to int if it is not NaN
    if household["occupancy"] is not None and not isnan(household["occupancy"]):
        household["occupancy"] = int(household["occupancy"])
    else:
        household["occupancy"] = None

    # Convert rental_units to int if it is not None and not NaN
    if household["rental_units"] is not None and not isnan(household["rental_units"]):
        household["rental_units"] = int(household["rental_units"])
    else:
        household["rental_units"] = None

    # Convert evs to int if it is not None and not NaN
    if household["EVs"] is not None and not isnan(household["EVs"]):
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
        consumption=consumption,
        house_size=household["house_size"],
        labeled=labeled
    )).scalar_one()

    return household_id


def get_or_create_device_id(conn: Connection, device: str, household_id: int, data: dict, daily_consumption: float,
                            event_consumption: float) -> int:
    """
    Get or create device ID if it doesn't exist yet
    ## Parameters
    `conn` : The connection to the database
    `device` : The device name
    `household_id` : The household ID
    `data` : The consumption data
    `daily_consumption` : The daily consumption
    `event_consumption` : The event consumption
    ## Returns
    `int` : The device ID
    """
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
        name=preprocess_string(device),
        loadprofile_daily=data[device]["daily"].flatten().astype(float).tolist(),
        loadprofile_weekly=data[device]["weekly"].flatten().astype(float).tolist(),
        loadprofile_monthly=data[device]["monthly"].flatten().astype(float).tolist(),
        daily_consumption=daily_consumption,
        event_consumption=event_consumption

    )).scalar_one()
    return device_id
