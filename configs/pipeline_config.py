from dotenv import load_dotenv
from os import environ

# load the .env file
load_dotenv()
# path to the raw data folder
RAW_DATA_PATH = "./data/raw/"

# path to the folder to save the parsed data
PARSED_DATA_PATH = "./data/parsed/"

# path to the folder to save the loadprofiles
LOADPROFILES_PATH = "./data/loadprofiles/"

# path to the folder containing metadata
METADATA_PATH = "./data/metadata/datasets/"

# path to the folder to save the generated metadata
GENERATED_METADATA_PATH = "./data/"

# path to the folder to save the consumption data
CONSUMPTION_DATA_PATH = "./data/"



# folder to save cleaned raw data with removed devices for training
TRAINING_DATA_CLEANED_FOLDER = "./data/training_data/raw/"

# path to the folder containing the trained model
MODEL_PATH = "./data/trained_models/InceptionTime/"

# path to the labels generated by generate_training_data.py
LABELS_PATH = "./data/training_data/labels.pkl"

# path to the folder to save the predicted appliances
PREDICTED_APPLIANCES_PATH = "./data/"

# endpoint to the knowledge graph where the data will be inserted
KNOWLEDGE_GRAPH_ENDPOINT = "http://193.2.205.14:7200/repositories/Electricity_Graph"

# postgres url to store the data
POSTGRES_URL = f"postgresql://{environ['DATABASE_USER']}:{environ['DATABASE_PASSWORD']}@193.2.205.14:5432/Energy"

# steps to be executed
STEPS = [
    "parse",
    "loadprofiles",
    "metadata",
    "consumption-data",
    "db-reset",
    "generate-links",
    "predict-devices",
    "add-predicted-devices"
]

# list of datasets to preprocess
DATASETS = [
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


# datasets on which to predict appliances
PREDICT_DATASETS = [
    "IDEAL",
    "LERTA"
]
