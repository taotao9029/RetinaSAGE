import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm


_DINO_MODEL_CANDIDATES = (
    "dinov2_vits16",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2",
    "vit_small_patch16_224.dino",
)


def _create_dino_backbone(img_size):
    available_models = set(timm.list_models())
    for model_name in _DINO_MODEL_CANDIDATES:
        if model_name in available_models:
            backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                img_size=img_size,
            )
            patch_size = backbone.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                if patch_size[0] != patch_size[1]:
                    raise RuntimeError(f"只支持方形 patch，当前 patch_size={patch_size}")
                patch_size = patch_size[0]
            return backbone, int(patch_size), model_name

    supported = ", ".join(sorted(name for name in available_models if "dino" in name.lower()))
    raise RuntimeError(
        "当前 timm 未找到可用的 DINO 模型。"
        f"可用 DINO 模型：{supported or '无'}；请升级 timm 或安装对应预训练模型。"
    )


# --------------------------
# 手写 GATv2 层（不依赖torch_geometric）
# --------------------------
class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.w = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, adj_mask):
        """
        x: [B, N, D_in]
        adj_mask: [B, N, N] 邻接矩阵 mask, True=相邻节点
        return: [B, N, heads * out_dim]
        """
        B, N, _ = x.shape
        h = self.w(x)  # [B,N,heads*D]
        h = rearrange(h, "b n (h d) -> b h n d", h=self.heads)
        # GATv2: a^T LeakyReLU(W [h_i || h_j])
        h_i = h.unsqueeze(3)
        h_j = h.unsqueeze(2)
        h_cat = h_i + h_j
        attn_score = self.leakyrelu(h_cat)
        attn_score = torch.einsum("bhnid,hd->bhni", attn_score, self.attn)

        # mask非邻接节点
        mask = adj_mask.unsqueeze(1)
        attn_score = attn_score.masked_fill(~mask, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        out = torch.matmul(attn_weight, h)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out

# --------------------------
# Patch Graph Builder：构建8邻接邻接矩阵，不生成edge_index
# --------------------------
class PatchGraphBuilder(nn.Module):
    def __init__(self, patch_size=16, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patch_h = img_size // patch_size
        self.num_patch_w = img_size // patch_size
        self.num_patches = self.num_patch_h * self.num_patch_w

        # 预计算patch坐标和8邻接mask
        coords = torch.stack(torch.meshgrid(torch.arange(self.num_patch_h), torch.arange(self.num_patch_w), indexing="ij"), dim=-1)
        coords = coords.reshape(-1, 2)
        dist = torch.cdist(coords.float(), coords.float(), p=1)
        self.register_buffer("adj_mask", dist <= 1) # 曼哈顿距离<=1，8邻域

    def forward(self, patch_feats):
        """
        patch_feats: [B, num_patches, feat_dim]
        return: node_feat [B,N,D], adj_mask [B,N,N]
        """
        B, num_patches, _ = patch_feats.shape
        if num_patches == self.num_patches:
            adj = self.adj_mask.unsqueeze(0).expand(B, -1, -1)
        else:
            side = math.isqrt(num_patches)
            if side * side != num_patches:
                raise ValueError(f"patch 数量必须是完全平方数，实际为 {num_patches}")
            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(side, device=patch_feats.device),
                    torch.arange(side, device=patch_feats.device),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            dist = torch.cdist(coords.float(), coords.float(), p=1)
            adj = (dist <= 1).unsqueeze(0).expand(B, -1, -1)
        return patch_feats, adj

# --------------------------
# GATv2 Spatial Encoder 逐帧空间图聚合
# --------------------------
class GATv2SpatialEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat_layer = GATv2Layer(in_dim, hidden_dim//heads, heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, adj_mask):
        gat_out = self.gat_layer(node_feat, adj_mask)
        gat_out = self.norm(gat_out)
        gat_out = self.dropout(gat_out)
        return gat_out

# --------------------------
# Causal GRU Temporal Encoder 【核心：单向GRU，严格因果】
# --------------------------
class CausalTemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, layers: int, heads: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if layers>1 else 0.0
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward_sequence(self, x: torch.Tensor, valid: torch.Tensor):
        """Return a causal hidden state for every time step."""
        B, T, N, D = x.shape
        seq = x.permute(0, 2, 1, 3)
        seq = seq * valid[:, None, :, None].to(dtype=seq.dtype)
        seq = seq.reshape(B * N, T, D)

        hidden_seq, _ = self.gru(seq)
        hidden_seq = self.norm_out(hidden_seq)
        return hidden_seq.reshape(B, N, T, D).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        """
        x: [B, T, N, D]
        valid: [B, T] bool mask，True代表有效帧
        return last_hidden: [B,N,D]
        """
        B, T, N, D = x.shape
        hidden_seq = self.forward_sequence(x, valid)
        last_index = valid.long().sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(B, device=x.device)
        return hidden_seq[batch_index, last_index]

# --------------------------
# V1VideoModel 完整主模型（类名不变）
# --------------------------
class V1VideoModel(nn.Module):
    def __init__(
        self,
        dino_feat_dim=384,
        hidden_dim=256,
        temporal_layers=2,
        gat_heads=4,
        dropout=0.1,
        num_classes=1,
        pred_future_dim=None,
        feature_dim=None,
        heads=None,
        graph="grid4",
        seed=None,
        no_gat=False,
    ):
        super().__init__()
        if feature_dim is not None:
            dino_feat_dim = int(feature_dim)
        if heads is not None:
            gat_heads = int(heads)
        if pred_future_dim is None:
            pred_future_dim = dino_feat_dim

        self.feature_dim = dino_feat_dim
        self.no_gat = no_gat
        self.dino_backbone = None
        self.dino_model_name = None
        graph_patch_size = 16
        if feature_dim is None:
            self.dino_backbone, dino_patch_size, dino_model_name = _create_dino_backbone(224)
            self.dino_model_name = dino_model_name
            for p in self.dino_backbone.parameters():
                p.requires_grad = False
            self.dino_backbone.eval()
            graph_patch_size = dino_patch_size

        self.graph_builder = PatchGraphBuilder(img_size=224, patch_size=graph_patch_size)
        self.gat_spatial = GATv2SpatialEncoder(dino_feat_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        self.no_gat_proj = nn.Linear(dino_feat_dim, hidden_dim) if no_gat else None
        self.global_feature_proj = nn.Linear(dino_feat_dim, hidden_dim)
        self.temporal = CausalTemporalEncoder(hidden_dim, layers=temporal_layers, heads=gat_heads, dropout=dropout)

        # 分类头
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        # 未来局部特征预测头
        self.future_pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_future_dim)
        )

    def _encode_spatial(self, frame_patch_feats):
        spat_feats = []
        for t in range(frame_patch_feats.shape[1]):
            node_x, adj = self.graph_builder(frame_patch_feats[:, t])
            if self.no_gat:
                g_feat = self.no_gat_proj(node_x)
            else:
                g_feat = self.gat_spatial(node_x, adj)
            spat_feats.append(g_feat)
        return torch.stack(spat_feats, dim=1)

    def _forward_cached_features(self, regions, global_features, valid_mask, return_details):
        if regions.ndim != 4:
            raise ValueError(f"缓存 regions 应为 [B,T,N,F]，实际为 {tuple(regions.shape)}")
        if global_features.ndim != 3:
            raise ValueError(
                f"缓存 global_features 应为 [B,T,F]，实际为 {tuple(global_features.shape)}"
            )
        if regions.shape[-1] != self.feature_dim or global_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"特征维度不匹配：模型需要 {self.feature_dim}，"
                f"regions={regions.shape[-1]}，global_features={global_features.shape[-1]}"
            )

        spat_seq = self._encode_spatial(regions)
        temporal_seq = self.temporal.forward_sequence(spat_seq, valid_mask)
        last_index = valid_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(regions.shape[0], device=regions.device)
        last_temporal_feat = temporal_seq[batch_index, last_index]
        last_global_feat = global_features[batch_index, last_index]

        global_feat = last_temporal_feat.mean(dim=1)
        global_feat = global_feat + self.global_feature_proj(last_global_feat)
        logits = self.class_head(global_feat.unsqueeze(-1)).squeeze(-1)

        # t 时刻只预测 t+1，目标与预测均为 [B,T-1,N,F]。
        future_pred = self.future_pred_head(temporal_seq[:, :-1])
        gt_future_feat = regions[:, 1:]
        state = last_temporal_feat.mean(dim=1)
        return logits, future_pred, gt_future_feat, state if return_details else None

    def _forward_video(self, video_frames, valid_mask, output_time=None):
        if self.dino_backbone is None:
            raise RuntimeError("当前模型由缓存特征模式创建，不能直接输入原始视频")

        """
        video_frames: [B, T, 3, 224, 224]
        valid_mask: [B, T] bool
        output_time: optional time index used to read a causal prefix output
        return logits, future_pred
        """
        B, T, C, H, W = video_frames.shape
        # 逐帧提取DINO patch特征，丢弃CLS token
        frame_patch_feats = []
        with torch.no_grad():
            for t in range(T):
                img = video_frames[:, t, :, :, :]
                features = self.dino_backbone.forward_features(img)
                if isinstance(features, dict):
                    patch_feat = features.get("x_norm_patchtokens")
                    if patch_feat is None:
                        patch_feat = features["x_prenorm"][:, 1:, :]
                else:
                    expected_patches = self.graph_builder.num_patches
                    if features.shape[1] == expected_patches + 1:
                        patch_feat = features[:, 1:, :] # [B,N,384]，丢弃CLS
                    elif features.shape[1] == expected_patches:
                        patch_feat = features
                    else:
                        raise RuntimeError(
                            f"DINO 模型 {self.dino_model_name} 输出 {features.shape[1]} 个 token，"
                            f"但图构建器需要 {expected_patches} 个 patch token"
                        )
                frame_patch_feats.append(patch_feat)
        frame_patch_feats = torch.stack(frame_patch_feats, dim=1) # [B, T, N, 384]

        spat_seq = self._encode_spatial(frame_patch_feats)

        # GRU因果时序
        temporal_seq = self.temporal.forward_sequence(spat_seq, valid_mask)
        if output_time is None:
            last_index = valid_mask.long().sum(dim=1).clamp_min(1) - 1
            batch_index = torch.arange(B, device=video_frames.device)
            last_temporal_feat = temporal_seq[batch_index, last_index]
        else:
            if not 0 <= output_time < T:
                raise ValueError(f"output_time must be in [0, {T}), got {output_time}")
            last_temporal_feat = temporal_seq[:, output_time] # [B,N,D]

        # 分类分支
        global_feat = last_temporal_feat.mean(dim=1) # [B,D]
        logits = self.class_head(global_feat.unsqueeze(-1))

        future_pred = self.future_pred_head(temporal_seq[:, :-1])
        return logits, future_pred

    def forward(
        self,
        inputs,
        valid_mask_or_global_features,
        feature_valid_mask=None,
        output_time=None,
        return_details=False,
    ):
        """支持原始视频模式和缓存特征模式。"""
        if inputs.ndim == 5:
            if feature_valid_mask is not None:
                raise ValueError("原始视频模式不应传入第三个位置参数")
            return self._forward_video(inputs, valid_mask_or_global_features, output_time)
        if inputs.ndim == 4:
            if feature_valid_mask is None:
                raise ValueError("缓存特征模式需要传入 valid mask")
            return self._forward_cached_features(
                inputs,
                valid_mask_or_global_features,
                feature_valid_mask,
                return_details,
            )
        raise ValueError(f"输入必须是视频 [B,T,C,H,W] 或缓存特征 [B,T,N,F]，实际 ndim={inputs.ndim}")

# --------------------------
# 因果不变性测试脚本（可单独运行）
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V1VideoModel().to(device)
    model.eval()

    torch.manual_seed(0)
    B = 1
    T = 4
    vid_A = torch.randn(B, T, 3, 224, 224, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    vid_B = vid_A.clone()
    vid_B[:,2:,:,:,:] = 0
    with torch.no_grad():
        logit_A, _ = model(vid_A, mask, output_time=1)
        logit_B, _ = model(vid_B, mask, output_time=1)

    diff = torch.max(torch.abs(logit_A - logit_B)).item()
    print(f"最大绝对误差: {diff:.8f}")
    if diff < 1e-5:
        print("✅ 因果测试通过！模型看不到未来帧")
    else:
        print("❌ 因果测试失败")
