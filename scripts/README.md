## Use

# Parser Script

This script executes a collection of parsers on multiple datasets and saves the results to a pickle file for each dataset and a merged file of all datasets.

## Usage

You can use the script from the command line with the following syntax:


`python parsers/run_parsers.py <data_path> <save_path>`


Where:
* `<data_path>` is the path to your raw data directory.
* `<save_path>` is the path to the directory where you want to save the parsed results.

For example:
`python parsers/run_parsers.py path/to/data/ path/to/save/`

This will run all the parsers on the datasets found in `path/to/data/` directory and save the parsed results in the `path/to/save/` directory.




## Output

The script runs the relevant parser for each dataset in the input directory. The parsed results are then saved in a pickle file in your specified save path. The data for each dataset is in the shape:

household : { appliance : {dataframe with timestamps and values in watts} }


## Supported Datasets

Currently, the script supports the following datasets:

* DEDDIAG
* DEKN
* DRED
* ECO
* ENERTALK
* HEART
* HES
* HUE
* IAWE
* IDEAL
* LERTA
* REDD
* REFIT
* SustDataED2
* SustData
* UCIML
* UKDALE

# Load Profile Script

This script processes multiple datasets and calculates the daily, weekly, and monthly load profiles.



`python loadprofiles.py <data_path> <save_path>`


Where:
* `<data_path>` is the path to your processed dataset folder.
* `<save_path>` is the path to the directory where you want to save the calculated load profiles.

For example:

`python loadprofiles.py path/to/data/ path/to/save/`


This will process datasets from `path/to/data/` directory and save the calculated load profiles in the `path/to/save/` directory.

## Output

The script generates load profiles for each device and aggregate consumption in each household in all the datasets. These are saved in a pickle file in your specified save path. In addition, a merged load profile across all datasets is also generated and saved.

# Generate metadata
`python generate_metadata.py <data_path> <save_path>`

Where:
* `<data_path>` is the path to your metadata datasets folder.
* `<save_path>` is the path to the directory where you want to save the metadata file.

## Output
This script generates metadata for households and stores it in a parquet file in the specified save path.

# Generate consumption data
This script generates consumption data for households and stores it in a pickle file in the specified save path.
`python generate_consumption_data.py <data_path> <save_path>`

Where:
* `<data_path>` is the path to the folder containing parsed datasets.
* `<save_path>` is the path to the directory where you want to save the consumption data file.

## Output
This script generates a pickle file containing average daily consumption in kWh and an average on/off event consumption in kWh for each household and each device and stores it in the specified save path.

# Reset database

This script will delete the database at the specified url in the enviroment variable DATABASE_URL .env file and recreate it with the provided data. Make sure you have the correct database url in the .env file and the data files in the correct location before running the script. 

`python database-reset.py <data_path> <loadprofiles_path> <consumption_data_path>`

* The metadata file should be named `residential_metadata.parquet` and should be in the `data_path` folder.
* The loadprofiles file should be named `merged_loadprofiles.pkl` and should be in the `loadprofiles_path` folder.
* The consumption data file should be named `consumption_data.pkl` and should be in the `consumption_data_path` folder.












