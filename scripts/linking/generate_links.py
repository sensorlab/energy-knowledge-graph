from SPARQLWrapper import SPARQLWrapper, JSON
from fuzzywuzzy import fuzz
from geopy.distance import geodesic
import math
import argparse



def get_wikidata_results(latitude : float, longitude : float):

    # wiki data query to get the cities in a 50km radius of the given coordinates
    query_wikidata = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>

    SELECT DISTINCT ?city ?cityLabel ?location
    WHERE {{
    SERVICE wikibase:around {{
        ?city wdt:P625 ?location .
        bd:serviceParam wikibase:center "Point({longitude} {latitude})"^^geo:wktLiteral .
        bd:serviceParam wikibase:radius "50" . 
    }}
    ?city wdt:P31/wdt:P279* wd:Q515 . 
    ?city rdfs:label ?cityLabel .
    FILTER(LANG(?cityLabel) = "en") .

    }} LIMIT 1000

    """
     
    sparql_wdata = SPARQLWrapper("https://query.wikidata.org/sparql")

    sparql_wdata.setQuery(query_wikidata)

    sparql_wdata.setReturnFormat(JSON)

    results = sparql_wdata.query().convert()
    return results

def query_wikidata_coordinates(latitude : float, longitude : float, label : str, data=None):
    """
    Query Wikidata for the coordinates of a city given its label and coordinates, if the label is not found, return the closest city
    Can be used with the results of a previous query to avoid querying Wikidata again by passing the results as the data parameter
    """
    if data is None:
        print("Querying Wikidata......")
        results = get_wikidata_results(latitude, longitude)
    else:
        results = data

        
    matched_cities = []
    # match the label of the city with the results and if the similarity is above 80% add it to the list of matches
    for r in results["results"]["bindings"]:
        ratio = fuzz.partial_ratio(r["cityLabel"]["value"], label)
        if ratio > 80:
            matched_cities.append((ratio, r))

    # if no match is found, return the closest city
    if len(matched_cities) == 0:
        min_distance = math.inf
        for r in results["results"]["bindings"]:
            coords = r["location"]["value"].split("(")[1].split(")")[0].split(" ")
            coords = (float(coords[1]), float(coords[0]))
            distance = geodesic((latitude, longitude), coords).kilometers
            if distance < min_distance:
                min_distance = distance
                closest_city = r
        return closest_city["city"]["value"]
    # sort the results by the ratio of the match and return the best match
    matched_cities.sort(key=lambda x: x[0], reverse=True)
    return matched_cities[0][1]["city"]["value"]



def query_graphDB_cities(endpoint : str):
    """Query the energy knowledge graph for the cities and their coordinates"""
    query_my_data = """
    PREFIX schema: <https://schema.org/>
    PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
    PREFIX data: <http://mydata.example.org/>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX voc: <http://vocabulary.example.org/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT DISTINCT ?cityName ?longitude ?latitude ?City
    WHERE {
    ?s rdf:type schema:Place .
    ?s schema:containedInPlace ?City .
    ?City rdf:type schema:City .
    ?City schema:name ?cityName .
    ?s schema:longitude ?longitude .
    ?s schema:latitude ?latitude . 
    }
    """
    # endpoint = "http://localhost:7200/repositories/test"

    sparlq_graphdb = SPARQLWrapper(endpoint)
    sparlq_graphdb.setQuery(query_my_data)
    sparlq_graphdb.setReturnFormat(JSON)
    results_Graphdb = sparlq_graphdb.query().convert()

    for r in results_Graphdb["results"]["bindings"]:
        print(r["cityName"]["value"], r["longitude"]["value"], r["latitude"]["value"])

    return results_Graphdb 

def query_graphdb_countries(endpoint):
    query_countries = """
    PREFIX voc: <http://vocabulary.example.org/>
    PREFIX saref: <https://saref.etsi.org/core/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX schema: <https://schema.org/>
    SELECT DISTINCT ?country WHERE {
    ?country rdf:type schema:Country . 
    
    }
    """
    
    sparql_graphdb = SPARQLWrapper(endpoint)
    sparql_graphdb.setQuery(query_countries)
    sparql_graphdb.setReturnFormat(JSON)

    results = sparql_graphdb.query().convert()
    uris = {}
    for uri in results["results"]["bindings"]:
        k =  uri["country"]["value"].split("/")[-1].replace("%20", " ")
        # special case for the US
        if k == "United States":
            k = "United States of America"
        uris[k] = uri["country"]["value"]
    return uris

def get_dbpedia_results(latitude : float, longitude : float):
    # dbpedia query to get the cities in a 50km radius of the given coordinates
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?city ?cityLabel 
    WHERE {{
      ?city a dbo:City .
      ?city rdfs:label ?cityLabel .
      ?city geo:lat ?lat .
      ?city geo:long ?long .
      FILTER (LANG(?cityLabel) = "en")
      FILTER (
        bif:st_intersects (
          bif:st_point (?long, ?lat),
          bif:st_point ({longitude}, {latitude}),
          50
        )
      )
    }}
    LIMIT 1000


    """

    sparql_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")

    sparql_dbpedia.setQuery(query)

    sparql_dbpedia.setReturnFormat(JSON)

    results = sparql_dbpedia.query().convert()

    return results

def query_dbpedia_coordinates(latitude : float, longitude : float, label : str, data=None):
    if data is None:
        print("Querying DBpedia......")
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?city ?cityLabel ?lat ?long
        WHERE {{
          ?city a dbo:City .
          ?city rdfs:label ?cityLabel .
          ?city geo:lat ?lat .
          ?city geo:long ?long .
          FILTER (LANG(?cityLabel) = "en")
          FILTER (
            bif:st_intersects (
              bif:st_point (?long, ?lat),
              bif:st_point ({longitude}, {latitude}),
              50
            )
          )
        }}
        LIMIT 1000


        """

        sparql_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")

        sparql_dbpedia.setQuery(query)

        sparql_dbpedia.setReturnFormat(JSON)

        results = sparql_dbpedia.query().convert()
    else:
        results = data

    matched_cities = []
    # check if the label of the city matches the label of the results and if the similarity is above 80% add it to the list of matches
    for r in results["results"]["bindings"]:
        ratio_partial = fuzz.partial_ratio(r["cityLabel"]["value"], label)
        ratio = fuzz.ratio(r["cityLabel"]["value"], label)

        ratio = (ratio + ratio_partial) / 2

        if ratio > 80:
            matched_cities.append((ratio, r))
    # if no match is found, return the closest city
    if len(matched_cities) == 0:
        min_distance = math.inf
        for r in results["results"]["bindings"]:
            coords = (float(r["lat"]["value"]), float(r["long"]["value"]))
            distance = geodesic((latitude, longitude), coords).kilometers
            if distance < min_distance:
                min_distance = distance
                closest_city = r

        return closest_city["city"]["value"]
    # sort the results by the ratio of the match and return the best match
    matched_cities.sort(key=lambda x: x[0], reverse=True)
    return matched_cities[0][1]["city"]["value"]
def query_wikidata_countries(country: str):
    """
    Query Wikidata for the country of a city given its city wikidata entity id
    """   
    # query_wikidata_countries =f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    # PREFIX wd: <http://www.wikidata.org/entity/>
    # PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    # PREFIX bd: <http://www.bigdata.com/rdf#>

    # SELECT ?city ?cityLabel ?country ?countryLabel
    # WHERE {{
    # wd:{city} rdfs:label ?cityLabel ;  # City (Montreal)
    #         wdt:P17 ?country .         # Country of the city
    # ?country rdfs:label ?countryLabel .
    
    # FILTER(LANG(?cityLabel) = "en" && LANG(?countryLabel) = "en").
    # }}
    # """
    query_wikidata_countries = f"""
    SELECT ?country WHERE {{
    ?country wdt:P31 wd:Q6256; # instance of a country
            rdfs:label "{country}"@en. # country name in English
    }}

    """
        
    sparql_wdata = SPARQLWrapper("https://query.wikidata.org/sparql")

    sparql_wdata.setQuery(query_wikidata_countries)

    sparql_wdata.setReturnFormat(JSON)

    results = sparql_wdata.query().convert()

    return results["results"]["bindings"][0]["country"]["value"]


def query_dbpedia_countries(country : str):
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>

    SELECT ?country WHERE {{
    ?country a dbo:Country ;
            rdfs:label "{country}"@en .
    }}
    LIMIT 1
    """
    sparql_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql_dbpedia.setQuery(query)
    sparql_dbpedia.setReturnFormat(JSON)
    results = sparql_dbpedia.query().convert()
    return results["results"]["bindings"][0]["country"]["value"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process energy graph data.")

    parser.add_argument("--energy_endpoint", default="http://193.2.205.14:7200/repositories/EnergyGraph_mixed", type=str, help="Sparql enpoint for the energy KG")
    parser.add_argument("--path_to_save", default="matches.nt", type=str, help="Path to save the matches")

    args = parser.parse_args() 

    # query the energy knowledge graph for the cities and their coordinates
    results_Graphdb_cities = query_graphDB_cities(args.energy_endpoint)
    result_Graphdb_countries = query_graphdb_countries(args.energy_endpoint)


    matches= []
    # iterate over the results and query Wikidata and DBpedia for each city
    for c in results_Graphdb_cities["results"]["bindings"]:
        label = c["cityName"]["value"]
        # if label != "Montreal":
        #     continue
        longitude = float(c["longitude"]["value"])
        latitude = float(c["latitude"]["value"])
        print(label, longitude, latitude)
        result_dbpedia = (c["City"]["value"], query_dbpedia_coordinates(latitude, longitude, label))
        result_wikidata = (c["City"]["value"], query_wikidata_coordinates(latitude, longitude, label))
        matches.append(result_dbpedia)
        matches.append(result_wikidata)
    from time import sleep
    sleep(5)
    # iterate over the results and query Wikidata for each country
    for c in result_Graphdb_countries:
        print(c)
        result_wikidata_countries = (result_Graphdb_countries[c], query_wikidata_countries(c))
        result_dbpedia_countries = (result_Graphdb_countries[c], query_dbpedia_countries(c))
        matches.append(result_wikidata_countries)
        matches.append(result_dbpedia_countries)

    triples = []
    # generate the triples for the matches
    for m in matches:
        s = "<"+str(m[0]) +"> " + "<http://www.w3.org/2002/07/owl#sameAs> " + "<"+str(m[1]) +"> .\n"
        print(s)
        triples.append(s)
    # save the triples
    with open(args.path_to_save, "w") as f:
        f.writelines(triples)