import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from typing import List, Any, Optional, Dict, Union, Tuple


# ---------------------------
# Utilities: convert networkx -> PyG Data
# ---------------------------
def convert_nx_to_pyg_data(
    graph: nx.DiGraph,
    txt_encoder_model,
    device: torch.device = torch.device("cpu")
) -> Data:
    nodes = list(graph.nodes())
    # collect texts in order
    node_texts = []
    for n in nodes:
        if isinstance(graph.nodes[n], dict) and 'label' in graph.nodes[n]:
            node_texts.append(graph.nodes[n]['label'])
        elif isinstance(graph.nodes[n], dict) and 'text' in graph.nodes[n]:
            node_texts.append(graph.nodes[n]['text'])
        else:
            node_texts.append(str(n))

    # Batch-encode all node texts (returns [N, bert_dim] or [bert_dim] if N==1)
    try:
        txt_embs = txt_encoder_model.encode(node_texts, batch_size=32, convert_to_tensor=True).clone().detach()  # torch.Tensor [N, H]
    except Exception as e:
        raise Exception(f'Can not encode. node_texts={node_texts}, '
                        f'Nodes:{list(graph.nodes())}. '
                        # f'First node type: {type(list(graph.nodes())[0])}'
                        f'')
    if txt_embs.dim() == 1:
        txt_embs = txt_embs.unsqueeze(0)

    # make sure bert_embs on device
    txt_embs = txt_embs.to(device)

    # pagerank 
    try:
        pagerank_scores = nx.pagerank(graph, max_iter=200)
    except Exception:
        pagerank_scores = {n: 1.0 / max(1, len(nodes)) for n in nodes}
    try:
        reverse_pagerank_scores = nx.pagerank(graph.reverse(copy=True), max_iter=200)
    except Exception:
        reverse_pagerank_scores = pagerank_scores.copy()

    x_list_input = []
    for i, n in enumerate(nodes):
        # structural features
        in_deg = float(graph.in_degree(n)) if graph.is_directed() else float(graph.degree(n))
        out_deg = float(graph.out_degree(n)) if graph.is_directed() else float(graph.degree(n))
        deg = float(graph.degree(n))
        pr = float(pagerank_scores.get(n, 0.0))
        rpr = float(reverse_pagerank_scores.get(n, 0.0))
        structural = torch.tensor([in_deg, out_deg, deg, pr, rpr], dtype=torch.float32, device=device)

        txt_emb = txt_embs[i]  # [H]
        x_list_input.append(torch.concat((txt_emb, structural)))

    x = torch.stack(x_list_input)

    # edges -> edge_index, edge_attr 
    edge_index = []
    edge_attr_list = []
    for u, v, data in graph.edges(data=True):
        ui = nodes.index(u)
        vi = nodes.index(v)
        edge_index.append([ui, vi])

        # --- Calculate a feature vector for each edge ---
        edge_centrality = nx.edge_betweenness_centrality(graph).get((u, v), 0.0)
        common_successors = len(list(nx.common_neighbors(nx.to_undirected(graph), u, v))) 
        
        # Simplified Jaccard for successors
        succ_u = set(graph.successors(u))
        succ_v = set(graph.successors(v))
        jaccard_out = len(succ_u.intersection(succ_v)) / max(1, len(succ_u.union(succ_v)))

        # ... TODO calculate other features ...

        # Create a feature vector for this edge
        feature_vector = [edge_centrality, common_successors, jaccard_out]
        edge_attr_list.append(feature_vector)

    if len(edge_attr_list) > 0:
        edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float32, device=device)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index_tensor = torch.empty((2,0), dtype=torch.long, device=device)
        # 3 =edge_centrality, common_successors, jaccard_out
        edge_attr_tensor  = torch.empty((0, 3), dtype=torch.float32, device=device)

    data = Data(x=x, edge_index=edge_index_tensor)
    data.edge_attr = edge_attr_tensor
    data.node_texts = node_texts
    data.node_ids=torch.arange(len(node_texts))
    return data


