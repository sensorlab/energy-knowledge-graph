import os
import sys
import argparse

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(project_root)
    sys.path.insert(0, project_root)
from src.linking.generate_links import generate_links
from src.run_parsers import parse_datasets
from src.loadprofiles import generate_loadprofiles
from src.generate_metadata import generate_metadata
from src.generate_consumption_data import generate_consumption_data
from src.database_reset import reset_database
from scripts.generate_training_data import generate_training_data
from src.remove_devices import remove_devices

from pathlib import Path
import gc

from configs import pipeline_config as config

if __name__ == "__main__":
    # get sample and full option from command line
    # if --sample is passed it uses only the sample data if --full is
    # passed it uses the full data
    parser = argparse.ArgumentParser(description='Process data for the energy knowledge graph')
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    parser.add_argument('--full', action='store_true', help='Use full data')
    args = parser.parse_args()

    if args.sample:
        datasets = [
            "REFIT",
            "ECO",
            "HES",
            "UKDALE",
            "HUE",
            "LERTA",
            "UCIML",
            "DEKN",
            "SUST1",
            "SUST2",
            "HEART",
            "ENERTALK",
            "DEDDIAG",
            "IDEAL"]
    elif args.full:
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
            "ECDUY",
            "PRECON",
            "EEUD"
        ]
    else:
        datasets = config.DATASETS

    # path to the raw data folder
    raw_data_path: Path = Path(config.RAW_DATA_PATH).resolve()

    # path to the folder to save the parsed data
    parsed_data_path: Path = Path(config.PARSED_DATA_PATH).resolve()
    if not parsed_data_path.exists():
        parsed_data_path.mkdir()

    # path to the folder to save the loadprofiles
    loadprofiles_path: Path = Path(config.LOADPROFILES_PATH).resolve()
    if not loadprofiles_path.exists():
        loadprofiles_path.mkdir()

    # path to the folder containing metadata
    metadata_path: Path = Path(config.METADATA_PATH).resolve()

    # path to the folder to save the generated metadata
    generated_metadata_path: Path = Path(config.GENERATED_METADATA_PATH).resolve()
    if not generated_metadata_path.exists():
        generated_metadata_path.mkdir()

    # path to the folder to save the consumption data
    consumption_data_path: Path = Path(config.CONSUMPTION_DATA_PATH).resolve()
    if not consumption_data_path.exists():
        consumption_data_path.mkdir()


    # path to the folder containing the pretrained model
    model_path: Path = Path(config.MODEL_PATH).resolve()

    #path to the labels
    labels_path: Path = Path(config.LABELS_PATH).resolve()

    # path to the folder to save the predicted appliances
    predicted_appliances_path: Path = Path(config.PREDICTED_APPLIANCES_PATH).resolve()
    if not predicted_appliances_path.exists():
        predicted_appliances_path.mkdir()

    # endpoint to the knowledge graph where the data will be inserted
    knowledge_graph_endpoint = config.KNOWLEDGE_GRAPH_ENDPOINT

    steps = config.STEPS

    # to avoid needing tensorflow to be installed when not using the "predict-devices" step
    if "predict-devices" in steps:
        from src.label_datasets import get_predicted_appliances

    if "add-predicted-devices" in steps:
        from src.add_predicted_devices import add_predicted_devices


    # datasets on which to predict appliances
    predict_datasets = config.PREDICT_DATASETS

    functions = {
        "parse": lambda: parse_datasets(raw_data_path, parsed_data_path, datasets),
        "loadprofiles": lambda: generate_loadprofiles(parsed_data_path, loadprofiles_path, datasets),
        "metadata": lambda: generate_metadata(metadata_path, generated_metadata_path, datasets),
        "consumption-data": lambda: generate_consumption_data(parsed_data_path, consumption_data_path, datasets),
        "db-reset": lambda: reset_database(generated_metadata_path / "residential_metadata.parquet",
                                           loadprofiles_path / "merged_loadprofiles.pkl",
                                           consumption_data_path / "consumption_data.pkl", datasets,
                                           config.POSTGRES_URL),
        "generate-links": lambda: generate_links(knowledge_graph_endpoint),
        "predict-devices": lambda: (
            get_predicted_appliances(parsed_data_path, model_path, labels_path, predicted_appliances_path,
                                     predict_datasets)),
        "add-predicted-devices": lambda: add_predicted_devices(predicted_appliances_path,
                                                               graph_endpoint=knowledge_graph_endpoint)

    }
    for step in steps:
        print("********************************************************************************************\n",
              "Starting step: ", step)
        functions[step]()
        gc.collect()
