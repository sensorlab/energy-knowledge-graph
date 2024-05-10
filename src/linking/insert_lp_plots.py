import SPARQLWrapper
from tqdm import tqdm

def query_households(endpoint):

    query_houses = """
    PREFIX saref: <https://saref.etsi.org/core/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX schema: <https://schema.org/>
    SELECT DISTINCT ?household WHERE {
    ?household rdf:type schema:House .
    ?household schema:name ?hname.
    LIMIT 10
    }
    """
    sparlq_graphdb = SPARQLWrapper(endpoint)
    sparlq_graphdb.setQuery(query_houses)
    sparlq_graphdb.setReturnFormat(JSON)

    try:
        results_Graphdb = sparlq_graphdb.query().convert()
        households = []
        for r in results_Graphdb["results"]["bindings"]:
            households.append(r["household"]["value"])
        return households
    except:
        print("Error in the query getting the households")
        return False
    
def insert_triples(endpoint, triples):
    sparlq_graphdb = SPARQLWrapper(endpoint)
    sparlq_graphdb.method = "POST"
    
    for t in tqdm(triples):
        query = f"""
        PREFIX saref: <https://saref.etsi.org/core/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX schema: <https://schema.org/>
        INSERT DATA {{
            {t}
        }}
        """
        sparlq_graphdb.setQuery(query)
        
        try:
            results_Graphdb = sparlq_graphdb.query()
        except:
            print("Error in the query inserting the triples")
            return False
    return True

if __name__ == "__main__":
    endpoint = "http://193.2.205.14:9999/blazegraph/namespace/energygraph/sparql"


    households = query_households(endpoint)
    triples = []
    for h in households:
        h_name = h.split("/")[-1]
        d_lp = f"<https://elkg.ijs.si/images/{h_name}_daily.png>"
        w_lp = f"<https://elkg.ijs.si/images/{h_name}_weekly.png>"
        m_lp = f"<https://elkg.ijs.si/images/{h_name}_monthly.png>"

        triple_d = f"<{h}> <https://elkg.ijs.si/ontology/hasDailyLoadprofile> {d_lp} ."
        triple_w = f"<{h}> <https://elkg.ijs.si/ontology/hasWeeklyLoadprofile> {w_lp} ."
        triple_m = f"<{h}> <https://elkg.ijs.si/ontology/hasMonthlyLoadprofile> {m_lp} ."

        triples.append(triple_d)
        triples.append(triple_w)
        triples.append(triple_m)

    print(insert_triples(endpoint, triples))