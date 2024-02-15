from run_parsers import parse_datasets
from loadprofiles import generate_loadprofiles
from generate_metadata import generate_metadata
from generate_consumption_data import generate_consumption_data
from database_reset import reset_database
import argparse
from pathlib import Path

if __name__ == "__main__":

    raw_data_path : Path = Path("./data/raw/").resolve()
    parsed_data_path : Path = Path("./data/parsed/").resolve()
    if not parsed_data_path.exists():
        parsed_data_path.mkdir()

    loadprofiles_path : Path = Path("./data/loadprofiles/").resolve()
    if not loadprofiles_path.exists():
        loadprofiles_path.mkdir()
    metadata_path : Path = Path("./data/metadata/datasets/").resolve()
    generated_metadata_path : Path = Path("./data/metadata/").resolve()
    consumption_data_path : Path = Path("./data/").resolve()

    steps = ["parse", "loadprofiles", "metadata", "consumption-data", "db-reset"]
        
    datasets = [
        "REFIT",
        "ECO",
        "HES",
        "UK-DALE",
        "HUE",
        "LERTA",
        "UCIML",
        "DRED",
        "REDD",
        "IAWE",
        "DEKN",
        "SUST1",
        "SUST2",
        "HEART",
        "ENERTALK",
        "DEDDIAG",
        "IDEAL",
        "ECD-UY"
    ]   

    functions = {
    "parse" : lambda: parse_datasets(raw_data_path, parsed_data_path, datasets),
    "loadprofiles": lambda: generate_loadprofiles(parsed_data_path, loadprofiles_path, datasets),
    "metadata": lambda: generate_metadata(metadata_path, generated_metadata_path, datasets) ,
    "consumption-data" : lambda: generate_consumption_data(parsed_data_path, consumption_data_path, datasets),      
    "db-reset" : lambda : reset_database(generated_metadata_path/"residential_metadata.parquet", loadprofiles_path/"merged_loadprofiles.pkl", consumption_data_path/"consumption_data.pkl")
    }
    for step in steps:
        functions[step]()

