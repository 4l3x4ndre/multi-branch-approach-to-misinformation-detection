import spacy
import networkx as nx
import hashlib
from matplotlib import pyplot as plt
from datetime import datetime

from numpy import true_divide
from pyvis.network import Network
from IPython.core.display import HTML, display_html

NODE_TYPES = {
  'ENTITY': ['PERSON', 'ORG', 'PRODUCT'],
  'EVENT': ['VERB', 'EVENT'],
  'STATE': ['STATE', 'CONDITION'],
  'LOCATION': ['GPE', 'LOC', 'FAC'],
  'TIME': ['DATE', 'TIME'],
  'ATTRIBUTE': ['ATTRIBUTE']
}

nlp = spacy.load("en_core_web_sm")

def generate_id(text):
  return hashlib.md5(text.lower().encode()).hexdigest()[:8]

def find_verb_sentences(doc):
  # Returns sentences with verbs (which we consider claims)
  claims = []
  for sent in doc.sents:
    if any(token.pos_ == "VERB" for token in sent):
      claims.append(sent)
  return claims

def create_nx_graph():
  return nx.MultiDiGraph()

def add_node_to_graph(G, node_id, node_type, text):
  G.add_node(node_id, type=node_type, text=text)

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
        # Update existing node attributes if needed
        merged.nodes[node].update(data)

    # Merge edges
    for u, v, data in G.edges(data=True):
      merged.add_edge(u, v, **data)

  return merged

def extract_relations(doc, G, node_map):
  def find_node_id(token):
    candidates = [
      token.text.lower(),
      token.text,
      f"the {token.text.lower()}",
      f"the {token.text}"
    ]
    for candidate in candidates:
      for key in node_map.keys():
        if candidate in key:
          return node_map[key]
    return None

  # Find main verbs
  for token in doc:
    if token.pos_ == "VERB":
      # EVENT node for verb
      verb_id = generate_id(token.text)
      add_node_to_graph(G, verb_id, "EVENT", token.text)

      # Find subject
      subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]

      for subj in subjects:
        subj_id = find_node_id(subj)
        if subj_id:
          if token.dep_ == "nsubjpass":
            add_edge_to_graph(G, verb_id, subj_id, "EXPERIENCES")
          else:
            add_edge_to_graph(G, subj_id, verb_id, "PERFORMS")

      # Find object
      objects = [child for child in token.children if child.dep_ in ("dobj", "pobj")]
      for obj in objects:
        obj_id = find_node_id(obj)
        if obj_id:
          add_edge_to_graph(G, verb_id, obj_id, "TARGETS")

      # Find locations (using prep phrases)
      for child in token.children:
        if child.dep_ == "prep" and child.text in ("in", "at", "on"):
          for loc in child.children:
            loc_id = find_node_id(loc)
            if loc_id:
              add_edge_to_graph(G, verb_id, loc_id, "LOCATED_IN")

    # Handle entity relationships (e.g., titles, organizations)
  for token in doc:
    if token.dep_ == "compound" and token.head.text in node_map:
      compound_text = f"{token.text} {token.head.text}"
      if compound_text in node_map:
        add_edge_to_graph(G, node_map[compound_text], node_map[token.head.text], "HAS_STATE")

      # Handle location relationships
    if token.dep_ == "prep" and token.text == "in":
      for child in token.children:
        if child.text in node_map and token.head.text in node_map:
          add_edge_to_graph(G, node_map[token.head.text], node_map[child.text], "LOCATED_IN")



def extract_single_claim(text):
  G = create_nx_graph()
  doc = nlp(text)

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
  doc = nlp(text)
  claims = find_verb_sentences(doc) # Assumption: claim sentences have verbs (can be greatly extended)

  graphs = []
  for claim in claims:
    graphs.append(extract_single_claim(claim.text))

  return merge_graphs(graphs)

def extract_claim_graph(text, batch_size=1000):
  if len(text) > batch_size: # Batch if text is longer than processing batch size
    chunks = []
    current_chunk = ""

    for sent in nlp(text).sents:
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


import webbrowser

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
    net.add_node(node, label=str(data["text"]), **data)

  for u, v, data in G.edges(data=True):
    label = data["type"]#data.get('type', '')
    net.add_edge(u, v, label=label, **data)

  net.from_nx(G)

  net.set_options(options)
  net.show('g.html')
  display_html(HTML('g.html'))
  webbrowser.open('g.html')
