import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import defaultdict
from itertools import combinations
import random
from torch.utils.data import DataLoader
import pandas as pd

class CellImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CellImageEncoder, self).__init__()

        # --- Initial Low-level Texture + Edge Detectors ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 64, 64] -> [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # --- Multi-Scale (Inception-like) Block ---
        self.conv_multi_scale = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=1, padding=0),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=5, padding=2),   # [B, 32, 64, 64]
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),   # fuse channels → [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # --- Residual Block with Attention ---
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, 64, 1, 1]
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )

        # --- Global Pooling and Projection to Embedding ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)   # → [B, 64, 1, 1]
        self.projector = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)  # low-level
        multi_scale_feats = [F.relu(conv(x)) for conv in self.conv_multi_scale]
        x = torch.cat(multi_scale_feats, dim=1)
        x = self.fusion(x)

        # Residual attention block
        residual = self.res_block(x)
        attn = self.attention(x)
        x = x + residual * attn

        # Global pooling and embedding
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, 64]
        x = self.projector(x)  # [B, embedding_dim]
        x = F.normalize(x, dim=1)  # normalize for contrastive learning
        return x

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)



def compute_class_separability(embeddings, labels, max_pairs=2000):
    """
    Compute class separability score = inter / intra
    Args:
        embeddings (Tensor): [N, D]
        labels (Tensor): [N]
        max_pairs (int): max number of pairs to use per category for memory efficiency
    Returns:
        dict with intra_class_dist, inter_class_dist, separability_score
    """
    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()
    class_to_embeddings = defaultdict(list)

    for emb, label in zip(embeddings, labels):
        class_to_embeddings[int(label.item())].append(emb)

    intra_dists = []
    inter_dists = []

    # Intra-class distances
    for emb_list in class_to_embeddings.values():
        if len(emb_list) < 2:
            continue
        pairs = list(combinations(emb_list, 2))
        random.shuffle(pairs)
        for emb1, emb2 in pairs[:max_pairs]:
            dist = torch.norm(emb1 - emb2).item()
            intra_dists.append(dist)

    # Inter-class distances
    class_items = list(class_to_embeddings.items())
    for i in range(len(class_items)):
        for j in range(i + 1, len(class_items)):
            emb_list_i = class_items[i][1]
            emb_list_j = class_items[j][1]
            sampled_i = random.sample(emb_list_i, min(len(emb_list_i), 10))
            sampled_j = random.sample(emb_list_j, min(len(emb_list_j), 10))
            for emb1 in sampled_i:
                for emb2 in sampled_j:
                    dist = torch.norm(emb1 - emb2).item()
                    inter_dists.append(dist)

    intra_mean = np.mean(intra_dists) if intra_dists else float('inf')
    inter_mean = np.mean(inter_dists) if inter_dists else 0.0
    score = inter_mean / (intra_mean + 1e-8)

    return {
        'intra_class_dist': intra_mean,
        'inter_class_dist': inter_mean,
        'separability_score': score
    }
