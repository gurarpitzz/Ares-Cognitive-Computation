r"""Social Graph Topology and Coordination Analysis (Section VII-C).

Implements:
1. Graph representation from directed edge lists / edge_index tensors.
2. Degree centrality: in-degree, out-degree, total degree, log degree ratio.
3. PageRank centrality: PR(u) = (1-d)/N + d \sum_{v \in N_{in}(u)} PR(v)/out(v).
4. Local clustering coefficients: C(u) = 2 e_u / (k_u (k_u - 1)).
5. Louvain community detection for modular social coordination structures.
6. Community-level bot density estimation with strict training-set label isolation (RF1, RF3).
7. Transductive vs inductive graph evaluation support (RF3).
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch


class SocialGraphExtractor:
    r"""Extracts topological and coordination features from social interaction graphs.
    
    Adheres strictly to Red-Flag Gates:
    - RF1: Training labels are used exclusively for computing community bot densities.
    - RF3: Supports both transductive (full graph topology visible, labels masked)
      and inductive (subgraph induced strictly on training nodes) topologies.
    """

    def __init__(self, inductive: bool = False, max_target_nodes: int = 11826):
        self.inductive = inductive
        self.max_target_nodes = max_target_nodes
        self.is_fitted = False

        self.full_graph = nx.DiGraph()
        self.target_graph = nx.Graph()
        self.pagerank_scores: Dict[int, float] = {}
        self.clustering_scores: Dict[int, float] = {}
        self.node_community_map: Dict[int, int] = {}
        self.community_sizes: Dict[int, int] = {}
        self.community_bot_densities: Dict[int, float] = {}
        self.feature_names = [
            "in_degree",
            "out_degree",
            "total_degree",
            "degree_ratio",
            "pagerank",
            "subgraph_degree",
            "clustering_coeff",
            "community_size",
            "community_bot_density",
        ]

    def fit(
        self,
        edge_index: Union[np.ndarray, torch.Tensor],
        train_indices: np.ndarray,
        train_labels: np.ndarray,
    ) -> "SocialGraphExtractor":
        r"""Build graph topology and compute centrality and community metrics.
        
        Args:
            edge_index: (2, E) array or tensor of directed edges [src, dst].
            train_indices: 1D array of node IDs belonging to training split.
            train_labels: 1D array of binary labels (0=human, 1=bot) for training split.
        """
        if isinstance(edge_index, torch.Tensor):
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
        else:
            src = edge_index[0]
            dst = edge_index[1]

        train_node_set = set(int(idx) for idx in train_indices)
        train_label_dict = {int(idx): int(lbl) for idx, lbl in zip(train_indices, train_labels)}

        # 1. Build Full Directed Graph (Transductive or Inductive)
        self.full_graph = nx.DiGraph()
        if self.inductive:
            # Only edges between training nodes
            edge_mask = np.isin(src, list(train_node_set)) & np.isin(dst, list(train_node_set))
            self.full_graph.add_nodes_from(train_node_set)
            self.full_graph.add_edges_from(zip(src[edge_mask], dst[edge_mask]))
        else:
            # Transductive: full topology is visible, but test labels remain unknown
            self.full_graph.add_nodes_from(range(self.max_target_nodes))
            self.full_graph.add_edges_from(zip(src, dst))

        # 2. Compute PageRank on full graph
        try:
            self.pagerank_scores = nx.pagerank(self.full_graph, alpha=0.85, max_iter=60, tol=1e-4)
        except Exception:
            # Fallback if power iteration fails to converge
            n = max(1, self.full_graph.number_of_nodes())
            self.pagerank_scores = {node: 1.0 / n for node in self.full_graph.nodes()}

        # 3. Build Target Subgraph for Community and Clustering Analysis
        self.target_graph = nx.Graph()
        if self.inductive:
            target_nodes = train_node_set
            mask = np.isin(src, list(target_nodes)) & np.isin(dst, list(target_nodes))
        else:
            target_nodes = set(range(self.max_target_nodes))
            mask = (src < self.max_target_nodes) & (dst < self.max_target_nodes)

        self.target_graph.add_nodes_from(target_nodes)
        self.target_graph.add_edges_from(zip(src[mask], dst[mask]))

        # 4. Local Clustering Coefficient on Target Subgraph
        self.clustering_scores = nx.clustering(self.target_graph)

        # 5. Louvain Community Detection
        communities = nx.community.louvain_communities(self.target_graph, seed=42)
        self.node_community_map = {}
        self.community_sizes = {}
        self.community_bot_densities = {}

        for comm_id, comm_nodes in enumerate(communities):
            self.community_sizes[comm_id] = len(comm_nodes)
            for node in comm_nodes:
                self.node_community_map[node] = comm_id

            # RF1: Calculate community bot density strictly from training nodes
            train_nodes_in_comm = [n for n in comm_nodes if n in train_label_dict]
            if train_nodes_in_comm:
                bot_count = sum(train_label_dict[n] for n in train_nodes_in_comm)
                self.community_bot_densities[comm_id] = float(bot_count) / len(train_nodes_in_comm)
            else:
                # Default to base rate in training set if no training nodes present
                self.community_bot_densities[comm_id] = float(np.mean(train_labels)) if len(train_labels) > 0 else 0.5

        self.is_fitted = True
        return self

    def extract_node_features(self, node_id: int) -> np.ndarray:
        """Extract 9-dimensional graph feature vector for a specific node."""
        if not self.is_fitted:
            raise RuntimeError("SocialGraphExtractor must be fitted before extracting features")

        in_deg = float(self.full_graph.in_degree(node_id)) if self.full_graph.has_node(node_id) else 0.0
        out_deg = float(self.full_graph.out_degree(node_id)) if self.full_graph.has_node(node_id) else 0.0
        tot_deg = in_deg + out_deg
        deg_ratio = float(np.log1p(in_deg) - np.log1p(out_deg))

        pr = float(self.pagerank_scores.get(node_id, 0.0))
        sub_deg = float(self.target_graph.degree(node_id)) if self.target_graph.has_node(node_id) else 0.0
        clustering = float(self.clustering_scores.get(node_id, 0.0))

        comm_id = self.node_community_map.get(node_id, -1)
        comm_size = float(self.community_sizes.get(comm_id, 1))
        comm_bot_density = float(self.community_bot_densities.get(comm_id, 0.5))

        return np.array([
            in_deg,
            out_deg,
            tot_deg,
            deg_ratio,
            pr,
            sub_deg,
            clustering,
            comm_size,
            comm_bot_density,
        ], dtype=float)

    def transform(self, node_indices: np.ndarray) -> np.ndarray:
        """Transform an array of node IDs into an (N, 9) feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("SocialGraphExtractor must be fitted first")

        features = [self.extract_node_features(int(idx)) for idx in node_indices]
        return np.array(features, dtype=float)
