import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, TransformerConv, GATv2Conv
import math

from corpus_truth_manipulation.config import CONFIG

from loguru import logger


class FeatureProjector(nn.Module):
  def __init__(self, text_dim, struct_dim, common_dim):
    super(FeatureProjector, self).__init__()

    self.text_proj = nn.Linear(text_dim, common_dim)
    self.struct_proj = nn.Linear(struct_dim, common_dim)

    self.alpha = nn.Parameter(torch.tensor(1.0))
    self.beta = nn.Parameter(torch.tensor(1.0))

  def forward(self, text_embeds, struct_features):
    proj_text = self.text_proj(text_embeds)
    proj_struct = self.struct_proj(struct_features)

    final_features = self.alpha * proj_text + self.beta * proj_struct
    return final_features

class EvidenceEncoder(torch.nn.Module):
  def __init__(self, in_channels, hidden_dims, out_channels, conv):
    super().__init__()
    self.projector = FeatureProjector(CONFIG.model.graph_text_encoder_dim, 5, 773)
    self.conv1 = conv(in_channels, hidden_dims//4, heads=4)
    self.conv2 = conv(hidden_dims, out_channels//2, heads=2)
    self.dropout = nn.Dropout(p=0.2)

  def forward(self, x, edge_index):
    node_features = self.projector(x[:, :CONFIG.model.graph_text_encoder_dim], x[:, CONFIG.model.graph_text_encoder_dim:])
    x = self.conv1(node_features, edge_index)
    x = F.relu(x)
    x = self.dropout(x)
    return self.conv2(x, edge_index)

class NodeScorer(torch.nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.scorer = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim//2),
      nn.ReLU(),
      nn.Dropout(p=0.1),
      nn.Linear(hidden_dim//2, 1)
    )

  def forward(self, node_embeddings):
    return self.scorer(node_embeddings)

class CrossGraphAttention(torch.nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.q = nn.Linear(hidden_dim, hidden_dim//2)
    self.k = nn.Linear(hidden_dim, hidden_dim//2)
    self.v = nn.Linear(hidden_dim, hidden_dim//2)
    self.layer_norm = nn.LayerNorm(hidden_dim//2)
    self.dropout = nn.Dropout(p=0.1)
    self.scaling = math.sqrt(hidden_dim//2)

  def forward(self, claim_nodes, evidence_nodes, claim_batch, evidence_batch):
    unique_batches = torch.unique(claim_batch)
    attended_outputs = []

    for batch_idx in unique_batches:
      batch_claim = claim_nodes[claim_batch == batch_idx]
      batch_evidence = evidence_nodes[evidence_batch == batch_idx]

      q = self.q(batch_claim)  # [num_claim_nodes, hidden_dim//2]
      k = self.k(batch_evidence)  # [num_evidence_nodes, hidden_dim//2]
      v = self.v(batch_evidence)  # [num_evidence_nodes, hidden_dim//2]

      # Attention scores: [num_claim_nodes, num_evidence_nodes]
      attention = torch.matmul(q, k.transpose(-2, -1)) / self.scaling
      attention_probs = F.softmax(attention, dim=-1)
      attention_probs = self.dropout(attention_probs)

      attended = torch.matmul(attention_probs, v)
      attended = self.layer_norm(attended)
      attended_outputs.append(attended)

    # Repack into a single tensor
    return torch.cat(attended_outputs, dim=0)



class ClaimVerifier(torch.nn.Module):
    def __init__(self, in_channels, int_dims=1024, hidden_dim=512, conv="GAT", classifier=True):
        super().__init__()

        if conv == "GAT":
          self.conv = GATConv
        elif conv == "GATv2":
          self.conv = GATv2Conv
        elif conv == "Transformer":
          self.conv = TransformerConv
        else:
          raise ValueError("Invalid conv type")

        self.encoder = EvidenceEncoder(in_channels, int_dims, hidden_dim, self.conv)
        print(self.conv)
        self.cross_attention = CrossGraphAttention(hidden_dim)
        self.node_scorer = NodeScorer(hidden_dim)
        self.classifier_flag = classifier

        combined_dim = hidden_dim + hidden_dim + hidden_dim//2  # evidence + claim + attended
        self.classifier = nn.Sequential(
          nn.Linear(combined_dim, 256),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(256, 1)
        )

    def forward(self, evidence_data, claim_data):
        evidence_embeddings = self.encoder(evidence_data.x, evidence_data.edge_index)
        claim_embeddings = self.encoder(claim_data.x, claim_data.edge_index)

        evidence_scores = self.node_scorer(evidence_embeddings)
        claim_scores = self.node_scorer(claim_embeddings)
        evidence_embeddings = evidence_embeddings * evidence_scores
        claim_embeddings = claim_embeddings * claim_scores

        attended_evidence = self.cross_attention(
            claim_embeddings,
            evidence_embeddings,
            claim_data.batch,
            evidence_data.batch
        )

        evidence_global = global_mean_pool(evidence_embeddings, evidence_data.batch)
        claim_global = global_mean_pool(claim_embeddings, claim_data.batch)
        attended_global = global_mean_pool(attended_evidence, claim_data.batch)

        combined = torch.cat([
        evidence_global,
        claim_global,
        attended_global
        ], dim=-1)

        if self.classifier_flag == False:
            return combined
        return torch.sigmoid(self.classifier(combined).squeeze())
