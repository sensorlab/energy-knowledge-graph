from run_parsers import parse_datasets
from loadprofiles import generate_loadprofiles
from generate_metadata import generate_metadata
from generate_consumption_data import generate_consumption_data
from database_reset import reset_database
from generate_training_data import generate_training_data
from src.remove_devices import remove_devices
from add_predicted_devices import add_predicted_devices
from label_datasets import get_predicted_appliances


import argparse
from pathlib import Path
import gc

if __name__ == "__main__":

    # path to the raw data folder
    raw_data_path : Path = Path("./data/raw/").resolve()

    # path to the folder to save the parsed data
    parsed_data_path : Path = Path("./data/parsed/").resolve()
    if not parsed_data_path.exists():
        parsed_data_path.mkdir()

    # path to the folder to save the loadprofiles
    loadprofiles_path : Path = Path("./data/loadprofiles/").resolve()
    if not loadprofiles_path.exists():
        loadprofiles_path.mkdir()

    # path to the folder containing metadata
    metadata_path : Path = Path("./data/metadata/datasets/").resolve()

    # path to the folder to save the generated metadata
    generated_metadata_path : Path = Path("./data/").resolve()
    if not generated_metadata_path.exists():
        generated_metadata_path.mkdir()
    
    # path to the folder to save the consumption data
    consumption_data_path : Path = Path("./data/").resolve()
    if not consumption_data_path.exists():
        consumption_data_path.mkdir()


    # folder to save training data
    training_data_folder: Path = Path("./data/training_data/").resolve()
    if not training_data_folder.exists():
        training_data_folder.mkdir()

    # folder to save cleaned raw data with removed devices
    training_data_cleaned_folder: Path = Path("./data/training_data/raw/").resolve()
    if not training_data_cleaned_folder.exists():
        training_data_cleaned_folder.mkdir()


    # path to the folder containing the trained model
    model_path : Path = Path("./data/trained_models/InceptionTime/").resolve()

    # path to the labels generated by generate_training_data.py
    labels_path : Path = Path("./data/training_data/labels.pkl").resolve()

    # path to the folder to save the predicted appliances
    predicted_appliances_path : Path = Path("./data/").resolve()
    if not predicted_appliances_path.exists():
        predicted_appliances_path.mkdir()

    # endpoint to the knowledge graph where the data will be inserted
    knowledge_graph_endpoint = "http://193.2.205.14:7200/repositories/Electricity_Graph"

    




    steps = [
            "parse",
            "loadprofiles",
            "metadata",
            "consumption-data",
            "db-reset",
            "training-data"
            "predict-devices",
            "add-predicted-devices"
    ]
        
    datasets = [
        "REFIT",
        "ECO",
        "HES",
        "UKDALE",
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
        "ECDUY"
    ]   
    # datasets used for training
    training_datsets = [
        "DEDDIAG",
        "DRED",
        "ECO",
        "ENERTALK",
        "HEART",
        "HES",
        "IAWE",
        "REDD",
        "REFIT",
        "UKDALE"
    ]
    # datasets on which to predict appliances
    predict_datasets = [
        "IDEAL",
        "LERTA"
    ]

    functions = {
    "parse" : lambda: parse_datasets(raw_data_path, parsed_data_path, datasets),
    "loadprofiles": lambda: generate_loadprofiles(parsed_data_path, loadprofiles_path, datasets),
    "metadata": lambda: generate_metadata(metadata_path, generated_metadata_path, datasets) ,
    "consumption-data" : lambda: generate_consumption_data(parsed_data_path, consumption_data_path, datasets),      
    "db-reset" : lambda : reset_database(generated_metadata_path/"residential_metadata.parquet", loadprofiles_path/"merged_loadprofiles.pkl", consumption_data_path/"consumption_data.pkl", datasets),
    "training-data" : lambda : (remove_devices(parsed_data_path, training_data_cleaned_folder, training_datsets), generate_training_data(training_data_cleaned_folder, training_data_folder, training_datsets)),
    "predict-devices" : lambda : get_predicted_appliances(parsed_data_path, model_path, labels_path, predicted_appliances_path, predict_datasets),
    "add-predicted-devices" : lambda : add_predicted_devices(predicted_appliances_path, graph_endpoint=knowledge_graph_endpoint)

    }
    for step in steps:
        functions[step]()
        gc.collect()

