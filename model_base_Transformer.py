from __future__ import annotations

import torch
from torch import nn

from model_Transformer import CausalTemporalTransformerEncoder, GATv2SpatialEncoder, PatchGraphBuilder


class BaseVideoModel(nn.Module):
    """仅使用缓存特征完成健康/脑卒二分类的基线模型。"""

    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        heads=4,
        temporal_layers=2,
        dropout=0.1,
        graph="grid4",
        no_gat=False,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.no_gat = no_gat
        self.graph_builder = PatchGraphBuilder(img_size=224, patch_size=16)
        self.gat_spatial = GATv2SpatialEncoder(
            self.feature_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.no_gat_proj = nn.Linear(self.feature_dim, hidden_dim) if no_gat else None
        self.global_feature_proj = nn.Linear(self.feature_dim, hidden_dim)
        self.temporal = CausalTemporalTransformerEncoder(
            hidden_dim,
            layers=temporal_layers,
            heads=heads,
            dropout=dropout,
        )
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_spatial(self, regions):
        spatial_features = []
        for time_index in range(regions.shape[1]):
            node_features, adjacency = self.graph_builder(regions[:, time_index])
            if self.no_gat:
                encoded = self.no_gat_proj(node_features)
            else:
                encoded = self.gat_spatial(node_features, adjacency)
            spatial_features.append(encoded)
        return torch.stack(spatial_features, dim=1)

    def forward(self, regions, global_features, valid, return_state=False):
        if regions.ndim != 4:
            raise ValueError(f"regions 应为 [B,T,N,F]，实际为 {tuple(regions.shape)}")
        if global_features.ndim != 3:
            raise ValueError(
                "global_features 应为 [B,T,F]，"
                f"实际为 {tuple(global_features.shape)}"
            )
        if regions.shape[-1] != self.feature_dim or global_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"特征维度不匹配：模型需要 {self.feature_dim}，"
                f"regions={regions.shape[-1]}，global_features={global_features.shape[-1]}"
            )

        spatial_sequence = self._encode_spatial(regions)
        temporal_sequence = self.temporal.forward_sequence(spatial_sequence, valid)
        last_index = valid.long().sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(regions.shape[0], device=regions.device)
        last_temporal = temporal_sequence[batch_index, last_index]
        last_global = global_features[batch_index, last_index]

        global_state = last_temporal.mean(dim=1)
        global_state = global_state + self.global_feature_proj(last_global)
        logits = self.class_head(global_state.unsqueeze(-1)).squeeze(-1)
        return (logits, global_state) if return_state else logits
