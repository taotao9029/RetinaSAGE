import torch
from torch import nn
from model_Transformer import CausalTemporalTransformerEncoder, GATv2SpatialEncoder, PatchGraphBuilder


class BaseVideoModel(nn.Module):
    """仅使用缓存特征完成健康/脑卒二分类的基线模型。
    Ablation控制：
        full(B6): no_gat=False, no_temporal=False, bi_temporal=False
        bi_temporal(B7): no_gat=False, no_temporal=False, bi_temporal=True
        no_temporal(B8): no_gat=False, no_temporal=True
        no_gat(B9): no_gat=True, no_temporal=False, bi_temporal=False
    """
    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        heads=4,
        temporal_layers=2,
        dropout=0.1,
        graph="grid4",
        no_gat=False,
        bi_temporal=False,
        no_temporal=False
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.no_gat = no_gat
        self.bi_temporal = bi_temporal
        self.no_temporal = no_temporal

        self.graph_builder = PatchGraphBuilder(img_size=224, patch_size=16)
        self.gat_spatial = GATv2SpatialEncoder(
            self.feature_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.no_gat_proj = nn.Linear(self.feature_dim, hidden_dim) if no_gat else None
        self.global_feature_proj = nn.Linear(self.feature_dim, hidden_dim)

        if not self.no_temporal:
            # 【方案2改动】不再传入 bidirectional 参数
            self.temporal = CausalTemporalTransformerEncoder(
                hidden_dim,
                layers=temporal_layers,
                heads=heads,
                dropout=dropout,
            )
        else:
            self.temporal = None

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

        # ========== B8分支：no_temporal，逐帧独立分类，最后取平均 ==========
        if self.no_temporal:
            B, T, N, D = spatial_sequence.shape
            logits_list = []
            for t in range(T):
                feat_t = spatial_sequence[:, t]
                feat_t = feat_t.mean(dim=1)
                global_t = global_features[:, t]
                state_t = feat_t + self.global_feature_proj(global_t)
                logit_t = self.class_head(state_t.unsqueeze(-1)).squeeze(-1)
                logits_list.append(logit_t)
            logits_stack = torch.stack(logits_list, dim=1)
            logits = logits_stack.mean(dim=1)
            return (logits, None) if return_state else logits

        # ========== B6/B7/B9：走时序Transformer，Base内部控制mask ==========
        B, T, N, D = spatial_sequence.shape
        # [B,T,N,D] -> [B*N, T, D]
        x = spatial_sequence.permute(0, 2, 1, 3).reshape(B * N, T, D)
        valid = valid.unsqueeze(1).repeat(1, N, 1).reshape(B * N, T)

        # 核心：根据 bi_temporal 决定是否启用因果mask
        if self.bi_temporal:
            # B7 双向，无因果mask，可以看到全部帧
            mask = None
        else:
            # B6/B9 单向因果mask，禁止偷看未来帧
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        src_key_padding_mask = ~valid

        # 调用原始 encoder，把mask和padding mask传进去
        out = self.temporal.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        # 还原维度 [B*N,T,D] -> [B,T,N,D]
        temporal_sequence = out.reshape(B, N, T, D).permute(0, 2, 1, 3)

        last_index = valid.reshape(B, N, T)[:, 0, :].long().sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(regions.shape[0], device=regions.device)
        last_temporal = temporal_sequence[batch_index, last_index]
        last_global = global_features[batch_index, last_index]
        global_state = last_temporal.mean(dim=1)
        global_state = global_state + self.global_feature_proj(last_global)
        logits = self.class_head(global_state.unsqueeze(-1)).squeeze(-1)
        return (logits, global_state) if return_state else logits
