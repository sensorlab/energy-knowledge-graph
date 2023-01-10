from sqlalchemy import text
from sqlalchemy.engine import Connection
import json


create_locations_table_sql = text('''
CREATE TABLE IF NOT EXISTS locations (
    id BIGSERIAL NOT NULL,
    meta JSONB,
    PRIMARY KEY (id)
)
''')

create_households_table_sql = text('''
CREATE TABLE IF NOT EXISTS households (
    id BIGSERIAL NOT NULL,
    lat FLOAT NOT NULL,
    lon FLOAT NOT NULL,
    parent_id BIGINT,
    location_id BIGINT,
    meta JSONB,
    PRIMARY KEY (id),
    FOREIGN KEY (parent_id) REFERENCES households (id) ON DELETE SET NULL,
    FOREIGN KEY (location_id) REFERENCES locations (id) ON DELETE SET NULL
)
''')

create_devices_table_sql = text('''
CREATE TABLE IF NOT EXISTS devices (
    id BIGSERIAL NOT NULL,
    household_id BIGINT NOT NULL,
    meta JSONB,
    PRIMARY KEY (id),
    FOREIGN KEY (household_id) REFERENCES households (id) ON DELETE CASCADE
)
''')




def ensure_tables(conn: Connection) -> None:
    conn.execute(create_locations_table_sql)
    conn.execute(create_households_table_sql)
    conn.execute(create_devices_table_sql)


#significant_fields = ('country', 'state', 'municipality', 'city', 'town', 'village', 'suburb', 'neighbourhood')

def get_or_create_location_id(conn: Connection, location:dict) -> int:
    """We will try to find area that exactly matches"""

    assert location.get('country', None), 'Country is mandatory'

    meta = location
    #meta = {}
    #for key in significant_fields:
    #    if location.get(key, None):
    #        meta[key] = location[key]


    # Returns 1 if exists, otherwise 0
    # Here we do exact comparison jsonb to jsonb
    query_location_sql = text('''
        SELECT id FROM locations WHERE meta = :meta
    ''')

    insert_location_sql = text('''
        INSERT INTO locations (meta)
        VALUES (:meta)
        RETURNING id;
    ''')

    meta = json.dumps(meta)

    #sql = query_location_sql.format(meta=json.dumps(meta))
    locations = conn.execute(query_location_sql, dict(meta=meta)).scalars().all()

    # Sanity checks
    assert len(locations) < 2, f'There are two locations with identical name: {locations}'

    # Does entry already exist? If it does, use it, otherwise create new one.
    if len(locations) > 0:
        location_id = locations[0]
        return location_id

    location_id = conn.execute(insert_location_sql, dict(meta=meta)).scalar_one()


    return location_id


def query_osm_metadata(lat:float, lon:float) -> dict:
    """OSM - OpenStreetMap"""
    import requests
    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&accept-language=en'
    res = requests.get(url)
    return res.json()


def get_or_create_household_id(conn: Connection, household: dict) -> int:
    """Currently there is no way to validate duplication of the entries."""

    insert_household_sql = text('''
        INSERT INTO households (lat, lon, parent_id, location_id, meta)
        VALUES (:lat, :lon, NULL, :location_id, :meta)
        RETURNING id;
    ''')

    # Pick geographical metadata about of household's location
    lat, lon = household['lat'], household['lon']
    address = query_osm_metadata(lat, lon)['address']

    # Obtain location ID
    location_id = get_or_create_location_id(conn, address)

    household_id = conn.execute(insert_household_sql, dict(lat=lat, lon=lon, location_id=location_id, meta='{}')).scalar_one()
    return household_id



def get_or_create_device_id(conn:Connection, device:dict, household_id:int) -> int:
    insert_device_sql = text('''
        INSERT INTO devices (household_id, meta)
        VALUES (:household_id, :meta)
        RETURNING id;
    ''')

    # Create entry in DB
    device_id = conn.execute(insert_device_sql, dict(household_id=household_id, meta=device)).scalar_one()
    return device_id
