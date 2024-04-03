## Use

# Parser Script

This script executes a collection of parsers on multiple datasets and saves the results to a pickle file for each dataset and a merged file of all datasets.

## Output

The script runs the relevant parser for each dataset specified. The parsed results are then saved in a pickle file in your specified save path. The data for each dataset is in the shape:

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
* PRECON
* EEUD
# Load Profile Script

This script processes multiple datasets and calculates the daily, weekly, and monthly load profiles.

## Output

The script generates load profiles for each device and aggregate consumption in each household in all the datasets. These are saved in a pickle file in your specified save path. In addition, a merged load profile across all datasets is also generated and saved.

# Generate metadata

## Output
This script generates a pickle file containing average daily consumption in kWh and an average on/off event consumption in kWh for each household and each device and stores it in the specified save path.

# Reset database

This script will delete the database at the specified url in the environment variable DATABASE_URL .env file and recreate it with the provided data. Make sure you have the correct database url in the .env file and the data files in the correct location before running the script. 

* The metadata file should be named `residential_metadata.parquet` and should be in the `data_path` folder.
* The loadprofiles file should be named `merged_loadprofiles.pkl` and should be in the `loadprofiles_path` folder.
* The consumption data file should be named `consumption_data.pkl` and should be in the `consumption_data_path` folder.




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

