import os
from tqdm import tqdm
import argparse
import shutil
import gc

# labelled datasets
from src.parsers.REFIT_parser import parse_REFIT
from src.parsers.ECO_parser import parse_ECO
from src.parsers.HES_parser import parse_HES
from src.parsers.UKDALE_parser import parse_UKDALE
from src.parsers.DRED_parser import parse_DRED
from src.parsers.DEKN_parser import parse_DEKN
from src.parsers.SUST2_parser import parse_SUST2
from src.parsers.HEART_parser import parse_HEART
from src.parsers.ENERTALK_parser import parse_ENERTALK
from src.parsers.DEDDIAG_parser import parse_DEDDIAG

# Mixed datasets
from src.parsers.IDEAL_parser import parse_IDEAL
from src.parsers.EEUD_parser import parse_EEUD

# Unlabelled datasets
from src.parsers.ECDUY_parser import parse_ECDUY
from src.parsers.HUE_parser import parse_HUE
from src.parsers.LERTA_parser import parse_LERTA
from src.parsers.UCIML_parser import parse_UCIML
from src.parsers.SUST1_parser import parse_SUST1
from src.parsers.PRECON_parser import parse_PRECON

from pathlib import Path

"""
This script runs the parsers on the corresponding data and saves the results to a pickle file for each dataset.
Usage: python run_parsers.py <path to data> <path to save folder>

The output data structure for each dataset has the following shape:

household: {appliance: {dataframe with timestamps as datetime index and values in watts in the first column}}

"""

def parse_datasets(data_path: Path, save_folder: Path, datasets: list[str]) -> None:
    """
    Save the parsed data to a pickle file for each dataset
    ### Parameters
    `data_path`: Path to the raw data
    `save_folder`: Path to the target folder for the pickle output.
    `datasets`: List of datasets to parse example: ["REFIT", "ECO"] will parse only REFIT and ECO
    """
    parse_functions = {
        "REFIT": parse_REFIT,
        "ECO": parse_ECO,
        "HES": parse_HES,
        "UKDALE": parse_UKDALE,
        "HUE": parse_HUE,
        "LERTA": parse_LERTA,
        "UCIML": parse_UCIML,
        "DRED": parse_DRED,
        "DEKN": parse_DEKN,
        "SUST1": parse_SUST1,
        "SUST2": parse_SUST2,
        "HEART": parse_HEART,
        "ENERTALK": parse_ENERTALK,
        "DEDDIAG": parse_DEDDIAG,
        "IDEAL": parse_IDEAL,
        "ECDUY": parse_ECDUY,
        "PRECON": parse_PRECON,
        "EEUD": parse_EEUD
    }
    


    for dataset in tqdm(os.listdir(data_path)):
        if dataset not in datasets:
            continue
        print(f"Processing {dataset}.... ")
        # avoid parsing with NILMTK and just copy the provided pickle files
        if dataset == "REDD":
            shutil.copy2(data_path / dataset / "REDD.pkl", save_folder / "REDD.pkl")
            continue
        if dataset == "IAWE":
            shutil.copy2(data_path / dataset / "IAWE.pkl", save_folder / "IAWE.pkl")
            continue
        # Get the appropriate parsing function from the dictionary
        parse_function = parse_functions.get(dataset)

        if parse_function:
            parse_function(data_path / dataset , save_folder / (dataset + ".pkl"))
        else:
            print(f"Dataset not found: {dataset}")
        gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Run parsers on energy data.")
    parser.add_argument("data_path", help = "Path to the raw data")
    parser.add_argument("save_folder", help = "Path to the save folder")
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    assert data_path.exists(), f"Path '{data_path}' does not exist!"

    save_folder = Path(args.save_folder).resolve()
    assert save_folder.exists(), f"Path '{save_folder}' does not exist!"

    # runs the parser on all the datasets comment out the ones you don't want to parse
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

    parse_datasets(data_path, save_folder, datasets)
