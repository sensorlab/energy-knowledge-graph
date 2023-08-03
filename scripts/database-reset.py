from dotenv import load_dotenv
from src.api import ensure_tables
from os import environ
import sqlalchemy as sa
from sqlalchemy import text ,inspect
from sqlalchemy.engine import Connection
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
from src.enrich_data import create_location_dict, create_weather_dict
from src.api import get_or_create_location_id, get_or_create_device_id, get_or_create_household_id

# Change path to data

# generated metadata from generate_metadata.py
DATA_PATH = './energy-knowledge-graph/data/metadata/residential_metadata.parquet'

# calculated loadprofiles from loadprofiles.py
LOADPROFILES_PATH = './energy-knowledge-graph/data/loadprofiles/merged_loadprofiles.pkl'

def load_data(conn:Connection):

    df = pd.read_parquet(DATA_PATH)
    loadprofiles = pd.read_pickle(LOADPROFILES_PATH)
    print("Populating database...")
    # iterate over rows in dataframe
    for _, row in df.iterrows():
        id = get_or_create_household_id(conn, row.to_dict())
        for device in loadprofiles[row['name']]:
            get_or_create_device_id(conn, device, id , loadprofiles[row['name']])
        
 



def main():
    load_dotenv()

    DATABASE_URL = environ['DATABASE_URL']
    assert DATABASE_URL, 'DATABASE_URL is required.'
    engine = sa.create_engine(DATABASE_URL, echo=False, future=True)

    if not database_exists(engine.url):
        print("Creating database...")
        create_database(engine.url)


    with engine.connect() as conn:
        # Cleanup existing tables
        conn.execute(text('DROP TABLE IF EXISTS devices CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS households CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS locations CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS weather CASCADE'))
        # save changes
        conn.commit()



    with engine.connect() as conn:
        # create tables
        ensure_tables(conn)
        conn.commit()

        # populate dataset
        load_data(conn)

        # save changes
        conn.commit()
        print("Done")

        
    
        

if __name__ == '__main__':
    main()