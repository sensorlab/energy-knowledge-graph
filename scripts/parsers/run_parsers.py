import os
from tqdm import tqdm
import sys

# TODO SPLIT IMPORTS INTO LABLLED AND UNLABELED
# import parsers
from REFIT_parser import parse_REFIT
from ECO_parser import parse_ECO
from HES_parser import parse_HES
from UKDALE_parser import parse_UKDALE
from HUE_parser import parse_HUE
from LERTA_parser import parse_LERTA
from SMART_parser import parse_SMART
from UCIML_parser import parse_UCIML
from DRED_parser import parse_DRED
from REDD_parser import parse_REDD
from IAWE_parser import parse_IAWE
from DEKN_parser import parse_DEKN
from SUST_parser import parse_SUST
from HEART_parser import parse_HEART
from ENERTALK_parser import parse_ENERTALK
from DEDDIAG_parser import parse_DEDDIAG
from IDEAL_parser import parse_IDEAL

# Unlabeled datasets
from ECDUY_parser import parse_ECDUY
"""
This script runs all the parsers on the data and saves the results to a pickle file for each dataset.
Usage: python run_parsers.py <path to data> <path to save folder>

the data for each dataset is in the shape

household : { appliance : {dataframe with timestamps and values in kWh} }
"""

if len(sys.argv) < 3:
    print("Usage: python run_parsers.py <path to data> <path to save folder>")
    sys.exit(1)
elif len(sys.argv) == 3:
    print("Processing data from " + sys.argv[1] + " and saving to " + sys.argv[2])
    data_path = sys.argv[1]
    save_folder = sys.argv[2]

#####################################################################
# TODO will be used to clean up the code later
# parse_functions = {
#     "SMART": parse_SMART,
#     "REFIT": parse_REFIT,
#     "ECO": parse_ECO,
#     "HES": parse_HES,
#     "UK-DALE": parse_UKDALE,
#     "HUE": parse_HUE,
#     "LERTA": parse_LERTA,
#     "UCIML": parse_UCIML,
#     "DRED": parse_DRED,
#     "REDD": parse_REDD,
#     "IAWE": parse_IAWE,
#     "DEKN": parse_DEKN,
#     "SUST": parse_SUST,
#     "HEART": parse_HEART
# }


# for dataset in tqdm(os.listdir(data_path)):
#     print(f"Processing {dataset}.... ")
    
#     # Get the appropriate parsing function from the dictionary
#     parse_function = parse_functions.get(dataset)
    
#     if parse_function:
#         parse_function(data_path + dataset + "/", save_folder + "/" + dataset + ".pkl")
#     else:
#         print(f"Dataset not found: {dataset}")

#########################################################################


# # path to all the data on the compute machine
# data_path = "../../shared/Energy_graph_datasets/raw/"

# # folder to save the preprocessed data
# save_folder = "../data/testing"


for dataset in tqdm(os.listdir(data_path)):
    print("Processing " + dataset + ".... ")
    if dataset == "SMART":
        # continue
        parse_SMART(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "REFIT":
        # continue
        parse_REFIT(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "ECO":
        # continue
        parse_ECO(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "HES":
        # continue
        parse_HES(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "UK-DALE":
        # continue
        parse_UKDALE(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "HUE":
        # continue
        parse_HUE(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "LERTA":
        # continue
        parse_LERTA(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "UCIML":
        # continue
        parse_UCIML(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "DRED":
        # continue
        parse_DRED(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "REDD":
        # continue
        parse_REDD(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "IAWE":
        # continue
        parse_IAWE(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "DEKN":
        # continue
        parse_DEKN(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "SUST":
        # continue
        parse_SUST(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "HEART":
        # continue
        parse_HEART(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "ENERTALK":
        # continue
        parse_ENERTALK(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "DEDDIAG":
        # continue
        parse_DEDDIAG(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "IDEAL":
        # continue
        parse_IDEAL(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    elif dataset == "ECD-UY":
        # continue
        parse_ECDUY(data_path + dataset+"/", save_folder + "/" + dataset + ".pkl")
    else:
        print("Dataset not found: " + dataset)
        # sys.exit(1)
