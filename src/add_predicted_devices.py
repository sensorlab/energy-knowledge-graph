from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from pathlib import Path
from tqdm import tqdm
def get_highest_device_id(endpoint="http://193.2.205.14:7200/repositories/Electricity_Graph")-> int:
    """
    Get the highest device id in the graph to know where to start adding new devices
    ## Parameters
    endpoint : The endpoint of the graph database
    ## Returns
    int : The highest device id



    """
    query_houses = """
    PREFIX voc: <http://vocabulary.example.org/>
    PREFIX saref: <https://saref.etsi.org/core/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX schema: <https://schema.org/>
    SELECT DISTINCT ?device WHERE {
    ?device rdf:type saref:Device .

    
    }
    """
    sparlq_graphdb = SPARQLWrapper(endpoint)
    sparlq_graphdb.setQuery(query_houses)
    sparlq_graphdb.setReturnFormat(JSON)
    

    try:
        results_Graphdb = sparlq_graphdb.query().convert()
        max_id = 0
        for r in results_Graphdb["results"]["bindings"]:
            id = int(r["device"]["value"].split("/")[-1])
            if id > max_id:
                max_id = id
        return max_id
    except:
        print("Error in the query")
        return False

def get_household(household : str, endpoint="http://193.2.205.14:7200/repositories/Electricity_Graph")-> str:
    """
    Get the household uri from the graph
    ## Parameters
    household : The name of the household
    endpoint : The endpoint of the graph database
    ## Returns
    str : The uri of the household
    """

    query_houses = f"""
    PREFIX voc: <http://vocabulary.example.org/>
    PREFIX saref: <https://saref.etsi.org/core/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX schema: <https://schema.org/>
    SELECT DISTINCT ?household WHERE {{
    ?household rdf:type schema:House .
    ?household schema:name ?hname.
    FILTER (?hname = '{household}')
    }}

    """
    sparlq_graphdb = SPARQLWrapper(endpoint)
    sparlq_graphdb.setQuery(query_houses)
    sparlq_graphdb.setReturnFormat(JSON)

    try:
        results_Graphdb = sparlq_graphdb.query().convert()
        if len(results_Graphdb["results"]["bindings"]) == 0:
            return False
        else:
            return results_Graphdb["results"]["bindings"]
    except:
        print("Error in the get household query")
        return False
    

def insert_device(device_id : int, device_name : str, household : str, endpoint="http://193.2.205.14:7200/repositories/Electricity_Graph"):
    """
    Insert a device into the graph
    ## Parameters
    device_id : The id of the device
    device_name : The name of the device
    household : The uri of the household
    endpoint : The endpoint of the graph database
    ## Returns
    bool : True if the device was inserted successfully, False otherwise
    """

    sparql_insert_query = f"""
    PREFIX voc: <http://vocabulary.example.org/>
    PREFIX saref: <https://saref.etsi.org/core/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX schema: <https://schema.org/>
    INSERT DATA {{
    <{household}> voc:containsDevice <http://mydata.example.org/public-devices/{device_id}> .
    <http://mydata.example.org/public-devices/{device_id}> rdf:type saref:Device .
    <http://mydata.example.org/public-devices/{device_id}> schema:name "{device_name}" .
    }}
    """
    sparlq_graphdb = SPARQLWrapper(endpoint +"/statements")
    sparlq_graphdb.setMethod('POST')
    sparlq_graphdb.setQuery(sparql_insert_query)

    try:
        results_Graphdb = sparlq_graphdb.query()
        return True
    except:
        print("Error in the insert query")
        return False

def add_predicted_devices(predicted_path : Path,  graph_endpoint : str):
    """

    Add the predicted devices to the graph
    ## Parameters
    predicted_path : The path to the predicted devices file
    graph_endpoint : The endpoint of the graph database
    ## Returns
    None
    """
    

    data = pd.read_pickle(predicted_path / "predicted_devices.pkl")

    start_id = get_highest_device_id(endpoint=graph_endpoint)
    for house in tqdm(data):
        household = get_household(house, endpoint=graph_endpoint)
        if household:
            household = household[0]["household"]["value"]
            for device in data[house]:
                start_id += 1
                insert_device(start_id, device, household, endpoint=graph_endpoint)
        else:
            print(f"House {house} not found in the graph")