import argparse
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy_utils import database_exists, create_database
from tqdm import tqdm

from src.api import ensure_tables
from src.api import get_or_create_device_id, get_or_create_household_id


# noinspection PyTypeChecker
def load_data(conn: Connection, df: pd.DataFrame, loadprofiles: dict, consumption: dict, datasets: list[str]) -> None:
    """
    Load the data into the database
    ### Parameters
    `conn` : SQLAlchemy Connection object
    `df` : DataFrame containing the metadata
    `loadprofiles` : Dictionary containing the loadprofiles
    `consumption` : Dictionary containing the consumption data
    `datasets` : List of datasets to process example: ["REFIT", "ECO"] will process only REFIT and ECO
    """

    print("Populating database...")
    # iterate over rows in dataframe
    for _, row in tqdm(df.iterrows()):
        if row['name'].split("_")[0] not in datasets:
            continue
        print(row['name'])
        # for debugging purposes
        # if "ECDUY" in row['name'] or "HUE" in row["name"] or "REFIT" in row["name"] or "UCIML" in row["name"] or "HES" in row["name"] or "ECO" in row["name"]or "LERTA" in row["name"] or "UKDALE" in row["name"] or "DRED" in row["name"]:
        #     continue
        if row['name'] not in loadprofiles:
            print("No loadprofile for: ", row['name'])
            continue
        # print(row['name'])
        labeled = False
        for d in loadprofiles[row['name']]:
            if d != "aggregate":
                labeled = True
                break

        id = get_or_create_household_id(conn, row.to_dict(), consumption[row['name']]["aggregate"]["daily"], labeled)
        for device in loadprofiles[row['name']]:
            if device == "aggregate":
                get_or_create_device_id(conn, device, id, loadprofiles[row['name']],
                                        consumption[row['name']][device]["daily"], None)
            else:
                get_or_create_device_id(conn, device, id, loadprofiles[row['name']],
                                        consumption[row['name']][device]["daily"],
                                        consumption[row['name']][device]["event"])


# noinspection PyShadowingNames
def reset_database(data_path: Path, loadprofiles_path: Path, consumption_data: Path, datasets: list[str],
                   db_url: str) -> None:
    """
    Reset the database specified in the .env file and populate it with data from the given paths

    ### Parameters
    `data_path` : Path to the residential_metadata.parquet file generated by generate_metadata.py
    `loadprofiles_path` : Path to the merged_loadprofiles.pkl file generated by loadprofiles.py
    `consumption_data` : Path to the consumption_data.pkl file generated by generate_consumption_data.py
    `datasets` : List of datasets to process example: ["REFIT", "ECO"] will process only REFIT and ECO

    """

    load_dotenv()

    # check if the paths are valid
    assert data_path.exists(), f"{data_path} does not exist."
    assert loadprofiles_path.exists(), f"{loadprofiles_path} does not exist."
    assert consumption_data.exists(), f"{consumption_data} does not exist."

    # load data
    df = pd.read_parquet(data_path)
    loadprofiles = pd.read_pickle(loadprofiles_path)
    consumption = pd.read_pickle(consumption_data)

    DATABASE_URL = db_url
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
        # save changes
        conn.commit()

    with engine.connect() as conn:
        # create tables
        ensure_tables(conn)
        conn.commit()

        # populate dataset
        load_data(conn, df, loadprofiles, consumption, datasets)

        # save changes
        conn.commit()
        print("Done")

