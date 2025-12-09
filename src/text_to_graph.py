import torch

import spacy
import networkx as nx
import hashlib
from typing import List

from loguru import logger
from networkx import MultiDiGraph

NODE_TYPES = {
  'ENTITY': ['PERSON', 'ORG', 'PRODUCT'],
  'EVENT': ['VERB', 'EVENT'],
  'STATE': ['STATE', 'CONDITION'],
  'LOCATION': ['GPE', 'LOC', 'FAC'],
  'TIME': ['DATE', 'TIME'],
  'ATTRIBUTE': ['ATTRIBUTE']
}

from src.utils.embeddings import NLP

def generate_id(text):
  return hashlib.md5(text.lower().encode()).hexdigest()[:8]

# REFINED VERSION OF find_verb_sentences:
def is_meaningful(sent):
    tokens = [t for t in sent if not t.is_punct and not t.is_stop]
    return len(tokens) >= 2 and any(t.pos_ in {"NOUN", "PROPN"} for t in tokens)
def find_claim_sentences(doc):
    """
    Identify sentences that are likely to contain a claim,
    including both verbal and nominal (verbless) structures.
    """
    claim_sents = []
    for sent in doc.sents:
        has_verb = any(t.pos_ == "VERB" for t in sent)
        has_nominal_predicate = any(t.pos_ in {"NOUN", "PROPN"} for t in sent) and any(
            t.pos_ in {"NOUN", "ADJ", "NUM"} for t in sent
        )

        # if "X Y" -> could be nominal claim (like "Paris capital")
        short_but_meaningful = len(sent) <= 6 and len(sent) > 1

        # Include sentences that have a verb OR seem like a nominal assertion
        if (has_verb or has_nominal_predicate or short_but_meaningful) and is_meaningful(sent):
            claim_sents.append(sent)

    return claim_sents

def create_nx_graph():
  return nx.MultiDiGraph()

def add_node_to_graph(G, node_id, node_type, text):
  G.add_node(node_id, type=node_type, text=text, label=text)

def add_edge_to_graph(G, source, target, edge_type, confidence=1.0,):
  G.add_edge(source, target, type=edge_type, confidence=confidence,)

def merge_graphs(graphs):
  # Merge multiple garphs into one
  if not graphs:
    return create_nx_graph()

  merged = graphs[0].copy()
  for G in graphs[1:]:
    for node, data in G.nodes(data=True):
      if node not in merged:
        merged.add_node(node, **data)
      else:
        # Update existing node attributes
        merged.nodes[node].update(data)

    # Merge edges
    for u, v, data in G.edges(data=True):
      merged.add_edge(u, v, **data)

  return merged

def extract_relations(doc, G, node_map):
    """
    Extracts relationships from the spaCy Doc and adds them to the graph.
    This version captures a wider range of grammatical dependencies for a richer graph.
    """
    # Helper to find a node ID for a given token by checking the noun chunk or entity it belongs to.
    def get_node_id_for_token(token):
        # Check if the token is part of a noun chunk or named entity that is already a node.
        # This links the relation to the full concept (e.g., "the powerful engine") not just one word ("engine").
        for chunk in doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return node_map.get(chunk.text.lower())
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return node_map.get(ent.text.lower())
        
        # If it's a verb, we use its base form (lemma) to generate an ID.
        if token.pos_ == 'VERB':
            return generate_id(token.lemma_)
            
        # Fallback for other tokens (like adjectives that become their own nodes).
        return node_map.get(token.text.lower())

    # Iterate through all tokens in the sentence to find relationships.
    for token in doc:
        # 1. Subject-Verb relationships (nsubj, nsubjpass)
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject_id = get_node_id_for_token(token)
            verb_lemma = token.head.lemma_
            verb_id = generate_id(verb_lemma)
            
            if subject_id:
                # Ensure the verb node exists in the graph.
                if not G.has_node(verb_id):
                    add_node_to_graph(G, verb_id, "EVENT", verb_lemma)
                
                # Define relationship type based on voice (active vs. passive).
                edge_type = "PERFORMS" if token.dep_ == "nsubj" else "EXPERIENCES"
                add_edge_to_graph(G, subject_id, verb_id, edge_type)

        # 2. Verb-Object relationships (dobj)
        elif token.dep_ == "dobj":
            object_id = get_node_id_for_token(token)
            verb_lemma = token.head.lemma_
            verb_id = generate_id(verb_lemma)

            if object_id:
                if not G.has_node(verb_id):
                    add_node_to_graph(G, verb_id, "EVENT", verb_lemma)
                add_edge_to_graph(G, verb_id, object_id, "TARGETS")

        # 3. Prepositional Phrases (pobj)
        elif token.dep_ == "pobj":
            pobj_id = get_node_id_for_token(token)
            preposition = token.head
            source_token = preposition.head
            source_id = get_node_id_for_token(source_token)
            
            if source_id and pobj_id:
                # The edge label becomes the preposition itself (e.g., "OVER", "WITH").
                edge_type = preposition.text.upper()
                if source_token.pos_ == 'VERB' and not G.has_node(source_id):
                     add_node_to_graph(G, source_id, "EVENT", source_token.lemma_)
                add_edge_to_graph(G, source_id, pobj_id, edge_type)

        # 4. Adjectival Modifiers (amod)
        elif token.dep_ == "amod":
            attribute_text = token.text
            attribute_id = generate_id(attribute_text)
            target_id = get_node_id_for_token(token.head)

            if target_id:
                if not G.has_node(attribute_id):
                    add_node_to_graph(G, attribute_id, "ATTRIBUTE", attribute_text)
                add_edge_to_graph(G, target_id, attribute_id, "HAS_ATTRIBUTE")

        # 5. Negation (neg) 
        elif token.dep_ == "neg":
            target_token = token.head
            if target_token.pos_ == "VERB":
                verb_id = generate_id(target_token.lemma_)
                if not G.has_node(verb_id):
                    add_node_to_graph(G, verb_id, "EVENT", target_token.lemma_)
                # Add a 'negated' attribute to the event node.
                G.nodes[verb_id]['negated'] = True

def extract_single_claim(text):
  G = create_nx_graph()
  doc = NLP(text)

  node_map = {}

  # Extract entities and create nodes
  for ent in doc.ents:
    node_type = next(
      (t for t, subtypes in NODE_TYPES.items()
      if ent.label_ in subtypes),
      'ENTITY'
    )

    node_id = generate_id(ent.text) # identifier for the node
    node_map[ent.text.lower()] = node_id
    node_map[ent.text] = node_id

    if not ent.text.startswith('the'):
      # Deduplicate: map nodes of type "the President" and "the president" to the same node id
      node_map[f"the {ent.text.lower()}"] = node_id
      node_map[f"the {ent.text}"] = node_id

    add_node_to_graph(G, node_id, node_type, ent.text)

  # Extract basic noun phrases not caught by NER
  for chunk in doc.noun_chunks:
    if chunk.text.lower() not in node_map:
      node_id = generate_id(chunk.text)
      node_map[chunk.text.lower()] = node_id
      node_map[chunk.text] = node_id
      add_node_to_graph(G, node_id, "ENTITY", chunk.text)

  # Extract relations
  extract_relations(doc, G, node_map)

  return G

def process_chunk(text):
  doc = NLP(text)
  claims = find_claim_sentences(doc) 

  graphs = []
  for claim in claims:
    graphs.append(extract_single_claim(claim.text))

  return merge_graphs(graphs)

def extract_claim_graph(text, batch_size=1000):
  if len(text) > batch_size: # Batch if text is longer than processing batch size
    chunks = []
    current_chunk = ""

    for sent in NLP(text).sents:
      if len(current_chunk) + len(sent.text) > batch_size:
        # If a sentence is longer than batch size, add it whole, otherwise concatenate multiple sentences
        chunks.append(current_chunk)
        current_chunk = sent.text
      else:
        current_chunk += " " + sent.text

    if current_chunk:
      chunks.append(current_chunk)

    # Process each chunk into graphs and merge them all
    graphs = []
    for chunk in chunks:
      graphs.append(process_chunk(chunk))
    return merge_graphs(graphs)

  return process_chunk(text)


def extract_claim_graphs(texts: List[str], batch_size=1000) -> tuple[list[MultiDiGraph], list[str]]:
    """
    Batch implementation to extract claim graphs from multiple texts.
    """
    all_graphs, all_texts = [], []
    for txt in texts:
        g = extract_claim_graph(txt, batch_size)
        if len(list(g.nodes())) > 0:
            all_texts.append(txt)
            all_graphs.append(g)
        else:
            all_graphs.append(torch.nan)
            all_texts.append(torch.nan)
    return all_graphs, all_texts


def create_node_features(graph, node_text, bert_model):
    """
    Create node features combining BERT embeddings and structural features.
    Combines:
    - BERT embedding (768-dim)
    - Structural features (5-dim): in-degree, out-degree, degree, PageRank, reverse PageRank
    Returns a tuple (bert_embedding, structural_features)
    """
    # Get BERT embedding (768-dim)
    bert_emb = bert_model.encode(node_text)
    
    # Calculate PageRank for the entire graph
    try:
        pagerank_scores = nx.pagerank(graph, max_iter=100)
        reverse_pagerank_scores = nx.pagerank(graph.reverse(), max_iter=100)
    except:
        # Handle edge cases (disconnected graphs, etc.)
        pagerank_scores = {node: 1.0/len(graph.nodes()) for node in graph.nodes()}
        reverse_pagerank_scores = pagerank_scores.copy()
    
    # Structural features (5-dim)
    structural_features = [
        float(graph.in_degree(node_text)),
        float(graph.out_degree(node_text)),
        float(graph.degree(node_text)),
        pagerank_scores.get(node_text, 0.0),
        reverse_pagerank_scores.get(node_text, 0.0)
    ]
    
    return bert_emb, structural_features


from pyvis.network import Network
from IPython.core.display import HTML, display_html


def visualize_graph(G):
    for _, properties in G.nodes.items():
        properties['size'] = 20

    options = '''
  const options = {
    "physics": {
      "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -4000,
          "centralGravity": 0.2,
          "springLength": 500,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      },
    "nodes": {
      "borderWidthSelected": 21,
      "font": {
        "size": 30,
        "face": "verdana"
      }
    }
  }
  '''
    net = Network(notebook=True,
                  cdn_resources='in_line',
                  height='1000px',
                  width='100%',
                  )

    for node, data in G.nodes(data=True):
        if 'text' in data:
            net.add_node(node, label=str(data["text"]), **data)
        else:
            net.add_node(node, label=str(data["label"]), **data)

    for u, v, data in G.edges(data=True):
        label = data.get('type', '')
        net.add_edge(u, v, label=label, **data)

    net.from_nx(G)

    net.set_options(options)
    net.show('g.html')
    display_html(HTML('g.html'))


if __name__ == '__name__':


    text = """THE CRUISE SHIP INDUSTRY\n\nCozumel is one of the world's largest cruise ship ports. Up to 7 ships visit each day, bringing over 3.6 million passengers to the island each year. While these tourists have become vital to the economy, this industry is the largest threat to the Cozumel coral reef ecosystem.\n\nTheir giant propellers disturb marine life, resulting in steep population declines\n\nbeneath their routes. They also stir up sediment which settle on the corals,\n\nblocking their photosynthesis, and starving them to death. Our site\n\nis frequently cloudly due to the passing ships and our volunteers need to\n\nconstantly dust the colonies to keep them healthy.\n\n\u200b\n\nShips carry pathogens and pests with their ballast waterShips use ballast water to maintain stability and maneuverability. This water is taken on in one port and discharged in another, often carrying with it a multitude of marine organisms, including potential pathogens. These pathogens can introduce new diseases or harmful algal blooms to the local marine environment, impacting both marine life and potentially human health.Stony Coral Tissue Loss Disease (SCTLD) was first observed in Florida, near Virginia Key in Miami-Dade County, in 2014. Ocean currents should have carried it up the coast towards New York. However, cruise ships brought SCTLD to Cozumel in 2018, against the ocean currents from Miami. In just a year, over 60% of Cozumel's corals died. Some coral species went entirely extinct in the wild.\n\n\u200b\n\nSimilarly, lionfish were first reported in South Florida waters in 1985.They are an invasive species from the Indo-Pacific. It was likely thataquarium owners released their pet fish into the wild. The fish has avoracious appetite and no natural predators here in the Carribean. We spearfish them whenever they are reported to preserve our fish life here.
    """

    graph = extract_claim_graph(text)
    visualize_graph(graph)
