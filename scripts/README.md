## Use

# Parser Script

This script executes a collection of parsers on multiple datasets and saves the results to a pickle file for each dataset and a merged file of all datasets.

## Usage

You can use the script from the command line with the following syntax:


`python parsers/run_parsers.py <data_path> <save_path>`


Where:
* `<data_path>` is the path to your dataset directory.
* `<save_path>` is the path to the directory where you want to save the parsed results.

For example:
`python parsers/run_parsers.py path/to/data/ path/to/save/`

This will run all the parsers on the datasets found in `path/to/data/` directory and save the parsed results in the `path/to/save/` directory.




## Output

The script runs the relevant parser for each dataset in the input directory. The parsed results are then saved in a pickle file in your specified save path. The data for each dataset is in the shape:

household : { appliance : {dataframe with timestamps and values in kWh} }


## Supported Datasets

Currently, the script supports the following datasets:

* SMART
* REFIT
* ECO
* HES
* UK-DALE
* HUE
* LERTA
* UCIML

# Load Profile Script

This script processes multiple datasets and calculates the daily, weekly, and monthly load profiles.



`python loadprofiles.py <data_path> <save_path>`


Where:
* `<data_path>` is the path to your dataset file.
* `<save_path>` is the path to the directory where you want to save the calculated load profiles.

For example:

`python loadprofiles.py path/to/data/ path/to/save/`


This will process datasets from `path/to/data/` directory and save the calculated load profiles in the `path/to/save/` directory.

## Output

The script generates load profiles for each device and aggregate consumption in each household in all the datasets. These are saved in a pickle file in your specified save path. In addition, a merged load profile across all datasets is also generated and saved.

# Generate metadata
`python generate_metadata.py <data_path> <save_path> [--save]`

Where:
* `<data_path>` is the path to your metadata datasets folder.
* `<save_path>` is the path to the directory where you want to save the metadata file.
* `--save` is an optional argument. If you include it, the metadata will be saved to the `<save_path>`; otherwise, the metadata will be generated but not saved.

# Reset database

This script will delete the database at the specified url in the enviroment variable DATABASE_URL .env file and recreate it with the data from the data folder. Make sure the data folder contains the generated metadata and the merged loadprofiles. 

* The metadata file should be named `residential_metadata.parquet` and should be in the `data/metadata` folder.
* The loadprofiles file should be named `merged_loadprofiles.pkl` and should be in the `data/loadprofiles` folder.

## Usage

`python database-reset.py`










