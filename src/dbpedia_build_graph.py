import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from itertools import islice
import networkx as nx
from src.text_to_graph import visualize_graph
from loguru import logger
from typing import List

from corpus_truth_manipulation.config import CONFIG

# --------------------------
# 1. Entity Linking
# --------------------------
def link_entities_spotlight(text, confidence=0.5, support=20):
    """
    Call DBpedia Spotlight API to extract entities from text.
    Returns a list of DBpedia URIs.
    """
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {"Accept": "application/json"}
    params = {"text": text, "confidence": confidence, "support": support}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  
        data = response.json()
        entities = []
        for resource in data.get("Resources", []):
            entities.append(resource["@URI"])
        return entities
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the DBpedia Spotlight API: {e}")
        return [] 

def link_to_label(string):
    """
    Simple heuristic to convert a DBpedia URI to a readable label.
    """
    if not string or not isinstance(string, str):
        logger.error(f"Invalid input to link_to_label: {string}")
    if 'http' in string:
        if "#" in string.split("/")[-1]:
            return string.split("#")[-1].replace("_", " ")
        return string.split("/")[-1].replace("_", " ")
    return string


# --------------------------
# 3. Claim → Merged Subgraph
# --------------------------
def build_claim_subgraph(claim_text, limit=50, confidence=0.5, support=20):
    """
    Given a claim string, link DBpedia entities and fetch a merged local subgraph.
    Adds node labels where available.
    Returns a single NetworkX DiGraph.
    """
    entities = link_entities_spotlight(claim_text, confidence=confidence, support=support)
    print(f"Linked entities: {entities}")

    G = nx.DiGraph()

    for ent in entities:
        subG = fetch_dbpedia_subgraph(ent, limit=limit)
        # Merge node attributes (esp. labels) when combining
        for n, attrs in subG.nodes(data=True):
            new_attrs = {}
            for k, v in attrs.items():
                if k == 'label':
                    new_attrs['text'] = link_to_label(v)
                elif k not in new_attrs:
                    new_attrs[k] = v
            if 'text' not in new_attrs:
                new_attrs['text'] = link_to_label(n)

            if n not in G.nodes:
                G.add_node(link_to_label(n), **new_attrs)

        # Merge edges (with predicate attribute)
        for u, v, attrs in subG.edges(data=True):
            attrs['predicate'] = link_to_label(attrs.get('predicate', 'relatedTo'))
            if not G.has_edge(u, v):
                G.add_edge(link_to_label(u), link_to_label(v), **attrs)


    return G

def build_claim_subgraph_batch(claim_texts, limit=50, confidence=0.5, support=20):
    """
    Given a list of claim strings, link DBpedia entities and fetch merged local subgraphs.
    Returns a list of NetworkX DiGraphs.
    """
    graphs = []
    for claim in claim_texts:
        G = build_claim_subgraph(claim, limit=limit, confidence=confidence, support=support)
        graphs.append(G)
    return graphs



# --------------------------
# 2. Local Subgraph Fetching
# --------------------------
def fetch_dbpedia_subgraph(entity_uri, limit=50, label_batch_size=3):
    """
    Fetch a local neighborhood (incoming + outgoing triples) from DBpedia.
    Returns a NetworkX DiGraph with node labels.
    """
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setMethod('POST')

    G = nx.DiGraph()

    # Outgoing edges
    sparql.setQuery(f"""
    SELECT ?p ?o WHERE {{
        <{entity_uri}> ?p ?o .
    }} LIMIT {limit}
    """)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        p = r["p"]["value"]
        o = r["o"]["value"]
        G.add_edge(entity_uri, o, predicate=p)

    # Incoming edges
    sparql.setQuery(f"""
    SELECT ?s ?p WHERE {{
        ?s ?p <{entity_uri}> .
    }} LIMIT {limit}
    """)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        s = r["s"]["value"]
        p = r["p"]["value"]
        G.add_edge(s, entity_uri, predicate=p)

    # Collect all nodes to label
    nodes_to_label = list(set(G.nodes) | {entity_uri})

    # --- Batch label queries ---
    # Only keep valid URIs for SPARQL VALUES
    uris_to_query = [n for n in nodes_to_label if n.startswith("http://") or n.startswith("https://")]

    labels = {}
    for batch in batch_iterable(uris_to_query, label_batch_size):
        values = " ".join(f"<{n}>" for n in batch)
        sparql.setQuery(f"""
        SELECT ?s ?label WHERE {{
            VALUES ?s {{ {values} }}
            ?s rdfs:label ?label .
            FILTER(lang(?label) = 'en')
        }}
        """)
        try:
            results = sparql.query().convert()
            for r in results["results"]["bindings"]:
                labels[r["s"]["value"]] = r["label"]["value"]
        except Exception as e:
            logger.warning(f"Warning fetching labels for batch {batch}: {e}")

    # Assign labels: use fetched label if available, else fallback
    for n in nodes_to_label:
        if n in labels:
            G.nodes[n]["label"] = labels[n]
        else:
            # fallback for literals / missing labels
            G.nodes[n]["label"] = n.split("/")[-1] if n.startswith("http") else n

    return G


def batch_iterable(iterable, batch_size):
    """Yield successive batches of size batch_size from iterable"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

def fetch_labels_for_uris(uris, batch_size=3):
    """
    Fetch English rdfs:label for a list of URIs in small batches.
    Returns a dict: {uri: label}
    """
    labels = {}
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setMethod('POST')

    for batch in batch_iterable(uris, batch_size):
        values = " ".join(f"<{u}>" for u in batch if u.startswith("http://") or u.startswith("https://"))
        query = f"""
        SELECT ?s ?label WHERE {{
            VALUES ?s {{ {values} }}
            ?s rdfs:label ?label .
            FILTER(lang(?label)='en')
        }}
        """
        sparql.setQuery(query)
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            labels[r["s"]["value"]] = r["label"]["value"]

    return labels

# @logger.catch 
def build_entity_graph_optimized(uris, neighbor_limit=50, batch_size_labels=10, label_cache={}):
    G = nx.DiGraph()
    uris = list(set(uris))

    # --- 1. Internal edges in batch ---
    values = " ".join(f"<{u}>" for u in uris)
    query = f"""
    SELECT ?s ?p ?o WHERE {{
        VALUES ?s {{ {values} }}
        ?s ?p ?o .
        FILTER(?o IN ({values}))
    }}
    """
    results = sparql_query(query)
    for r in results:
        G.add_edge(r['s'], r['o'], predicate=r['p'])

    # --- 2. External neighbors with labels ---
    for uri in uris:
        query = f"""
        SELECT ?p ?o ?label WHERE {{
            <{uri}> ?p ?o .
            OPTIONAL {{ ?o rdfs:label ?label FILTER(lang(?label)='en') }}
        }} LIMIT {neighbor_limit}
        """
        results = sparql_query(query)
        for r in results:
            G.add_edge(uri, r['o'], predicate=r['p'])
            if r.get('label'):
                G.nodes[r['o']]['label'] = r['label']

    # --- 3. Fill missing labels from cache or batch query ---
    nodes_missing_label = [n for n in G.nodes if 'label' not in G.nodes[n]]
    if nodes_missing_label:
        # only query nodes not in cache
        nodes_to_query = [n for n in nodes_missing_label if n not in label_cache]
        new_labels = fetch_labels_for_uris(nodes_to_query, batch_size=batch_size_labels)
        label_cache.update(new_labels)
        for n in nodes_missing_label:
            G.nodes[n]['label'] = label_cache.get(n, n.split("/")[-1])

    return G


import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from cachetools import LRUCache
from sklearn.metrics.pairwise import cosine_similarity

# Create a global SPARQL endpoint object (can reuse for multiple calls)
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
sparql.setMethod('POST')

def sparql_query(query):
    """
    Execute a SPARQL query and return a list of dicts with variable bindings.
    Example:
      [{'s': 'http://dbpedia.org/resource/Alice', 'p': '...', 'o': '...'}, ...]
    """
    sparql.setQuery(query)
    results = sparql.query().convert()
    bindings = results['results']['bindings']
    
    # convert to simple dict with values only
    out = []
    for b in bindings:
        entry = {k: v['value'] for k, v in b.items()}
        out.append(entry)
    return out

embedding_cache = LRUCache(maxsize=CONFIG.kg_label_cache_max_size)  # Embeddings of any URI/label
label_cache = LRUCache(maxsize = CONFIG.kg_label_cache_max_size)  # global cache across batches

def get_embedding(text: str, nlp_model):
    """
    Return embedding vector for a text, using FastText with caching.
    """
    if text in embedding_cache:
        return embedding_cache[text]
    vec = nlp_model(text).vector
    embedding_cache[text] = vec
    return vec
def semantic_top_neighbors_for_batch(batch_entities, batch_neighbors, nlp_model, k=5):
    """
    Compute top-k semantically relevant neighbors for one batch item.
    batch_entities: list of URIs (input entities)
    batch_neighbors: list of dicts {'o': uri, 'label': str, 'p': str}
    """
    # Compute average embedding of input entities (labels)
    entity_vecs = []
    labels = []
    for e in batch_entities:
        label = label_cache.get(e, e.split('/')[-1])
        labels.append(label)
        entity_vecs.append(get_embedding(label, nlp_model))
    if not entity_vecs:
        return []

    avg_vec = np.mean(entity_vecs, axis=0, keepdims=True)

    # Collect neighbor embeddings
    neighbor_vecs = []
    valid_neighbors = []
    labels_n = []
    for n in batch_neighbors:
        label = n.get('label') or n['o'].split('/')[-1]
        labels_n.append(label)
        neighbor_vecs.append(get_embedding(label, nlp_model))
        valid_neighbors.append(n)

    if not neighbor_vecs:
        return []

    sims = cosine_similarity(avg_vec, np.stack(neighbor_vecs))[0]
    top_idx = np.argsort(sims)[::-1][:k]

    top_neighbors = [valid_neighbors[i] for i in top_idx]
    return top_neighbors

def fetch_neighbors_for_uris(uris, neighbor_limit=50, max_workers=8):
    """Fetch DBpedia neighbors in parallel for a list of URIs"""
    def fetch(uri):
        query = f"""
        SELECT ?p ?o ?label WHERE {{
            <{uri}> ?p ?o .
            OPTIONAL {{ ?o rdfs:label ?label FILTER(lang(?label)='en') }}
        }} LIMIT {neighbor_limit}
        """
        try:
            return uri, sparql_query(query)
        except Exception as e:
            print(f"[WARN] Failed fetching neighbors for {uri}: {e}")
            return uri, []

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, u): u for u in uris}
        for f in as_completed(futures):
            uri, neighbors = f.result()
            results[uri] = neighbors
    return results

def build_entity_graph_from_names_batch(entities_list, nlp_model, 
                                        neighbor_limit=50, k_neighbors=10,
                                        batch_size_labels=10, max_workers=8)->List[nx.DiGraph]:
    """
    Build subgraphs for a batch of entity lists.
    Returns list of NetworkX DiGraphs.
    """
    graphs = []

    # Step 0: link entity names to URIs
    batch_uris = []
    for entities in entities_list:
        uris = []
        for n in entities:
            linked = link_entities_spotlight(n)
            if linked:
                uris.append(linked[0])
        uris = list(set(uris))
        batch_uris.append(uris)

    # Step 1: build internal edges for all batch URIs
    all_uris = set(u for uris in batch_uris for u in uris)
    all_uris_list = list(all_uris)
    uris_only = [u for u in all_uris_list if u.startswith("http://") or u.startswith("https://")]
    literals_only = [u for u in all_uris_list if not u.startswith("http://")]
    values_for_values = " ".join(f"<{u}>" for u in uris_only)
    values_for_values += " " + " ".join(f'"{u}"' for u in literals_only)
    filter_for_uris = ", ".join(f"<{u}>" for u in uris_only)

    query_internal = f"""
    SELECT ?s ?p ?o WHERE {{
        VALUES ?s {{ {values_for_values} }}
        ?s ?p ?o .
        FILTER(?o IN ({filter_for_uris}))
    }}
    """
    internal_edges = sparql_query(query_internal)

    # Step 2: external neighbors fetched in parallel
    def fetch_neighbors(uri):
        query = f"""
        SELECT ?p ?o ?label WHERE {{
            <{uri}> ?p ?o .
            OPTIONAL {{ ?o rdfs:label ?label FILTER(lang(?label)='en') }}
        }} LIMIT {neighbor_limit}
        """
        results = sparql_query(query)
        return uri, results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_neighbors, uri): uri for uri in all_uris_list}
        external_results = {}
        for f in as_completed(futures):
            uri, neighbors = f.result()
            external_results[uri] = neighbors

    # Step 3: build graphs per batch item
    for uris in batch_uris:
        G = nx.DiGraph()
        # internal edges
        for r in internal_edges:
            if r['s'] in uris or r['o'] in uris:
                G.add_edge(r['s'], r['o'], predicate=r['p'])
        # external neighbors
        for uri in uris:
            if uri not in G:
                G.add_node(uri)
            for r in external_results.get(uri, []):
                G.add_edge(uri, r['o'], predicate=r['p'])
                if 'label' in r and r['label']:
                    G.nodes[r['o']]['label'] = r['label']
        # fill missing labels from cache or batch fetch
        nodes_missing = [n for n in G.nodes if 'label' not in G.nodes[n]]
        nodes_to_query = [n for n in nodes_missing if n not in label_cache]
        if nodes_to_query:
            new_labels = fetch_labels_for_uris(nodes_to_query, batch_size=batch_size_labels)
            label_cache.update(new_labels)
        for n in nodes_missing:
            G.nodes[n]['label'] = label_cache.get(n, n.split('/')[-1])
        graphs.append(G)

    # Retrieve all uris from neighbors batch-wise:
    batch_uris_neighbors = []
    for uris in batch_uris:
        current_neighbors_uris = set()
        for uri in uris:
            current_neighbors_uris.update(r['o'] for r in external_results.get(uri, []))
        batch_uris_neighbors.append(list(current_neighbors_uris))

    external_results_batched = []
    for entity_uris in batch_uris:
        neighbors = []
        for uri in entity_uris:
            neighbors.extend(external_results.get(uri, []))
        external_results_batched.append(neighbors)

    # Step 4: Semantic expansion per batch item
    for i, uris in enumerate(batch_uris_neighbors):
        G = graphs[i]
            
        # Select top-k semantically coherent neighbors
        top_neighbors = semantic_top_neighbors_for_batch(uris, external_results_batched[i], nlp_model, k=k_neighbors)
            
        # Fetch their DBpedia neighbors in parallel
        top_neighbor_uris = [n['o'] for n in top_neighbors]
        expanded_neighbors = fetch_neighbors_for_uris(top_neighbor_uris, neighbor_limit, max_workers)

        # Add top neighbors + their fetched neighbors to the graph
        for n in top_neighbors:
            s = n.get('s', None) or n['o']
            o = n['o']
            p = n['p']

            # Add fetched neighbors for this top node
            for r in expanded_neighbors.get(o, []):
                G.add_edge(o, r['o'], predicate=r['p'])
                if 'label' in r and r['label']:
                    G.nodes[r['o']]['label'] = r['label']
                else:
                    G.nodes[r['o']]['label'] = label_cache.get(r['o'], r['o'].split('/')[-1])

        graphs[i] = G


    return graphs

# --------------------------
# Example
# --------------------------
if __name__ == "__main__":
    claim = "Barack Obama was born in Hawaii."
    G = build_claim_subgraph(claim, limit=20)
    print(f"Final subgraph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    for n, d in list(G.nodes(data=True))[:5]:
        print("Node:", n, "| Label:", d.get("text"))

    for u, v, d in list(G.edges(data=True))[:5]:
        print("Edge:", u, "--", d["predicate"], "-->", v)

    visualize_graph(G)
