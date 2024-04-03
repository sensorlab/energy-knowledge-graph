# Energy Knowledge Graph

The projects builds knowledge graph for energy consumption.[TODO expand add pipeline image?]

# File structure

- `data` - contains the raw datasets and their metadata and later preprocessed data
- `scripts` - contains the scripts for preprocessing the data and populating the database
- `notebooks` - contains notebooks for data exploration and model training(TODO may remove)
- `src` - contains model training and evaluation code and some helper scripts
- `requirements.txt` - contains the python dependencies for the project without tensorflow
- `requirements.tensorflow.txt` - contains the python dependencies for tensorflow



## Installation / Use

The scripts are described in more detail in the [scripts README](src/README.md).


1. Start a terminal and clone this [repository](https://github.com/sensorlab/energy-knowledge-graph) `git clone https://github.com/sensorlab/energy-knowledge-graph`
2. Navigate into energy-knowledge-graph directory, enter conda or virtualenv, and install dependecies with `pip install -r requirements.txt` for data preprocessing 
and `pip install -r requirements.tensorflow.txt --extra-index-url https://pypi.nvidia.com` if you want to use the machine learning part of the pipeline
3. Unzip the `data_dump_full.tar.gz` file in the data directory using `tar -xvf data_dump_full.tar.gz -C data`, optionally you can use only the data_sample.tar.gz file `tar -xvf data_sample.tar.gz -C data` instead 
4. Make sure that `./data/`folder contains contains required datasets and metadata data folder should be of structure `./data/metadata/` containing metadata and `./data/raw/` containing raw datasets
5. Create an .env file in the [scripts](./scripts/) directory with the following content:
    ```bash
    DATABASE_USER=<username to access PostgreSQL database>
    DATABASE_PASSWORD=<password to access PostgreSQL database>
    ```
6. Check [pipeline_config.py](configs/pipeline_config.py) for the configuration of the pipeline leave as is for default configuration
7. Run `python scripts/process_data.py` by default this will preprocess the datasets defined in [pipeline_config.py](configs/pipeline_config.py) and store it in the database if we pass the command line argument `--sample` it will preprocess only the datasets present in the sample dump and if we pass `--full` it will preprocess all the datasets present in the full dump
8. Access PostgreSQL at port `:5433`



## Detailed pipeline script usage

The pipeline can be customized by changing the configuration in the [pipeline_config](configs/pipeline_config.py) file.

In the [pipeline_config.py](configs/pipeline_config.py) file you can set the following parameters:





- `STEPS` - list of data processsing steps to be executed
- `DATASETS` - list of datasets to be preprocessed
- `TRAINING_DATASETS` - list of datasets to be used to generate the training data
- `PREDICT_DATASETS` - list of unlabelled datasets to run the pretrained model on
- `POSTGRES_URL` - the url for the postgres database to store the data
- various paths for where to store the data and where to read the data from this is explained in more detail in the [pipeline_config](configs/pipeline_config.py) file


The pipeline contains the following data processing steps:

1. [parse](src/run_parsers.py) - This script runs the parsers for the datasets and stores the parsed datasets in pickle files
2. [loadprofiles](src/loadprofiles.py) - This script calcualtes the load profiles for the households
3. [metadata](src/generate_metadata.py) - This script generates metadata for the households and stores it in a dataframe as a parquet file
4. [consumption-data](./scripts/consumption_data.py) - This script calculates the electrictiy consumption data for the households and their appliances
5. [db-reset](./scripts/db_reset.py) - This script resets and populates the database with the households metadata, load profiles and consumption data
6. [training-data](scripts/generate_training_data.py) - This script generates the training data for on/off appliance classification from the training datasets

and the following steps for predicting devices using a pretrained model (requires tensorflow):

1. [predict-devices](src/label_datasets.py) - This script predicts the devices for the households using a pretrained model
2. [add_predicted_devices](src/add_predicted_devices.py) - This script adds the predicted devices to the knowledge graph





# SPARQL examples

We provide some example SPARQL queries that can be run on the knowledge graph. We also host a SPARQL endpoint at [TODO link] where you can test your queries.


## Example 1: Query all countries with GDP greater than 50000

```sparql
PREFIX voc: <http://vocabulary.example.org/>
PREFIX saref: <https://saref.etsi.org/core/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <https://schema.org/>
SELECT ?gdp ?location ?countryName WHERE {
  ?location rdf:type schema:Place . 
  ?location voc:hasGDPOf ?gdp .
  ?location schema:containedInPlace ?country .
  ?country rdf:type schema:Country .
  ?country schema:name ?countryName .
  FILTER(?gdp > 50000) .
} 

```


## Example 2: Query all devices in a house with name "LERTA_4"

```sparql
PREFIX voc: <http://vocabulary.example.org/>
PREFIX saref: <https://saref.etsi.org/core/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <https://schema.org/>
SELECT DISTINCT ?house ?devices ?deviceNames ?houseName WHERE {
 ?house rdf:type schema:House .
 ?house schema:name ?houseName .
 ?house voc:containsDevice ?devices .
 ?devices schema:name ?deviceNames .
 FILTER(?houseName = "LERTA_4").

} 

```

## Example 3: Query household "UKDALE_1" and the city it is in as well as the corresponding city in dbpedia and wikidata

```sparql
PREFIX voc: <http://vocabulary.example.org/>
PREFIX saref: <https://saref.etsi.org/core/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <https://schema.org/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT DISTINCT ?houseName ?city ?dbpediaCity ?wikidataCity WHERE {
  ?house rdf:type schema:House .
  ?house schema:name ?houseName .
  ?house schema:containedInPlace ?place .
  ?place schema:containedInPlace ?city .
  ?city rdf:type schema:City .
  
  OPTIONAL {
    ?city owl:sameAs ?linkedCity .
    FILTER(STRSTARTS(STR(?linkedCity), "http://dbpedia.org/resource/"))
    BIND(?linkedCity AS ?dbpediaCity)
  }
  
  OPTIONAL {
    ?city owl:sameAs ?linkedCity2 .
    FILTER(STRSTARTS(STR(?linkedCity2), "http://www.wikidata.org/entity/"))
    BIND(?linkedCity2 AS ?wikidataCity)
  }
  
  FILTER(?houseName = "UKDALE_1")
}
    
```


[TODO add bibtex reference for paper]