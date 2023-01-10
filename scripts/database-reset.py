from dotenv import load_dotenv
from src.api import ensure_tables
from os import environ
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy_utils import database_exists, create_database
import pandas as pd

from src.api import get_or_create_location_id, get_or_create_device_id, get_or_create_household_id

def load_hue_dataset(conn:Connection):
    df = pd.read_parquet('./data/residential-metadata.parquet')

    df.drop(6, inplace=True) # Drop household without location information

    for _, row in df.iterrows():
        get_or_create_household_id(conn, row.to_dict())




def main():
    load_dotenv()

    DATABASE_URL = environ['DATABASE_URL']
    assert DATABASE_URL, 'DATABASE_URL is required.'
    engine = sa.create_engine(DATABASE_URL, echo=False, future=True)

    if not database_exists(engine.url):
        create_database(engine.url)


    with engine.connect() as conn:
        # Cleanup existing tables
        conn.execute(text('DROP TABLE IF EXISTS devices'))
        conn.execute(text('DROP TABLE IF EXISTS households'))
        conn.execute(text('DROP TABLE IF EXISTS locations'))

    with engine.connect() as conn:
        ensure_tables(conn)

        load_hue_dataset(conn)


    with engine.begin() as conn:
        print('Locations:', conn.execute(text('SELECT COUNT(1) FROM locations')).scalar_one())
        print('Households:', conn.execute(text('SELECT COUNT(1) FROM households')).scalar_one())

        print('Locations:', conn.execute(text('SELECT * FROM locations')).all())


if __name__ == '__main__':
    main()