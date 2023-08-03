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

def load_data(conn:Connection):
    # TODO change path

    df = pd.read_parquet('./energy-knowledge-graph/data/metadata/residential_metadata.parquet')
    loadprofiles = pd.read_pickle('./energy-knowledge-graph/data/merged_loadprofiles.pkl')

    for _, row in df.iterrows():
        # if "HUE" in row["name"] or "REFIT" in row["name"] or "UCIML" in row["name"]:
        #     continue
        print(row['name'])
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

        inspector = inspect(engine)


    with engine.connect() as conn:
        # create tables
        ensure_tables(conn)
        conn.commit()

        # populate dataset
        load_data(conn)


        conn.commit()

        inspector = inspect(engine)
        print(inspector.get_table_names())

        


    # with engine.begin() as conn:
    #     print('Locations:', conn.execute(text('SELECT COUNT(1) FROM locations')).scalar_one())
    #     print('Households:', conn.execute(text('SELECT COUNT(1) FROM households')).scalar_one())

    #     print('Locations:', conn.execute(text('SELECT * FROM locations')).all())

    #     print('Weather:', conn.execute(text('SELECT * FROM weather')).all())
    #     # print all tables of db
        

if __name__ == '__main__':
    main()