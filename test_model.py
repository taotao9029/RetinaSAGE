import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- GATv2 模块（和model_v1保持完全一致） --------------------------
class GATv2(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // heads

        self.lin_q = nn.Linear(in_dim, self.head_dim * heads)
        self.lin_k = nn.Linear(in_dim, self.head_dim * heads)
        self.lin_v = nn.Linear(in_dim, self.head_dim * heads)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.proj_out = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x: [B, N, C] N=patch数量
        B, N, _ = x.shape
        q = self.lin_q(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.lin_k(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.lin_v(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # GATv2 注意力计算
        attn_score = F.leaky_relu(q.unsqueeze(2) + k.unsqueeze(3))
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.out_dim)
        out = self.proj_drop(out)
        out = self.proj_out(out)
        return out

# -------------------------- 因果GRU时序模块（和model_v1保持完全一致） --------------------------
class CausalGRU(nn.Module):
    def __init__(self, hidden_dim, time_layers, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=time_layers,
            batch_first=True,
            bidirectional=False,  # 单向，严格因果，看不到未来帧
            dropout=dropout if time_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        # x: [B, T, C]
        out, _ = self.gru(x)
        out = self.dropout(out)
        return out

# -------------------------- Base模型主体 --------------------------
class BaseVideoModel(nn.Module):
    def __init__(
        self,
        feature_dim=384,
        hidden_dim=128,
        gat_heads=4,
        time_layers=2,
        dropout=0.1,
        ablation="full"  # full:使用GATv2; no_gat:patch均值池化
    ):
        super().__init__()
        self.ablation = ablation
        # Patch特征映射
        self.patch_proj = nn.Linear(feature_dim, hidden_dim)

        if self.ablation == "full":
            self.gatv2 = GATv2(in_dim=hidden_dim, out_dim=hidden_dim, heads=gat_heads, dropout=dropout)
        else:
            self.gatv2 = None

        # 因果时序编码器（单向GRU）
        self.temporal_encoder = CausalGRU(hidden_dim=hidden_dim, time_layers=time_layers, dropout=dropout)

        # 分类头（仅输出视频级二分类logits）
        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2)
        )

    def get_temporal_hidden(self, regions, global_feat, valid_mask):
        """
        主干前向函数，【主干输出，不进入分类头】
        用于和V1做主干一致性校验，返回时序隐状态 [B, T, hidden_dim]
        regions: [B, T, N_patch, feature_dim] DINO提取的patch特征
        global_feat: [B, T, feature_dim] 全局帧特征（本版主干暂时未使用，预留接口）
        valid_mask: [B, T] bool 有效帧mask
        """
        B, T, N, C = regions.shape
        patch_feat = self.patch_proj(regions) # [B,T,N,hidden_dim]

        frame_feats = []
        for t in range(T):
            frame_patch = patch_feat[:, t, :, :] # [B,N,hidden_dim]
            if self.ablation == "full":
                graph_feat = self.gatv2(frame_patch) # GAT空间建模
                graph_feat = torch.mean(graph_feat, dim=1) # patch维度全局聚合 [B, hidden_dim]
            else:
                # no_gat分支：patch直接均值
                graph_feat = torch.mean(frame_patch, dim=1)
            frame_feats.append(graph_feat)

        frame_seq = torch.stack(frame_feats, dim=1) # [B, T, hidden_dim]
        temporal_hidden = self.temporal_encoder(frame_seq, valid_mask)
        return temporal_hidden

    def forward(self, regions, global_feat, valid_mask):
        """
        模型主前向：输入视频patch特征，输出二分类logits
        return: logits [B,2]
        """
        temporal_hidden = self.get_temporal_hidden(regions, global_feat, valid_mask)
        last_hidden = temporal_hidden[:, -1, :] # 取最后时间步隐状态做视频分类
        logits = self.class_head(last_hidden)
        return logits

# -------------------------- 简易测试代码（独立验证Base模型可运行） --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseVideoModel(
        feature_dim=384,
        hidden_dim=128,
        gat_heads=4,
        time_layers=2,
        dropout=0.1,
        ablation="full"
    ).to(device)

    # 模拟输入
    B, T, N_patch = 2, 6, 16
    regions = torch.randn(B, T, N_patch, 384).to(device)
    global_feat = torch.randn(B, T, 384).to(device)
    valid_mask = torch.ones(B, T, dtype=torch.bool).to(device)

    with torch.no_grad():
        logits = model(regions, global_feat, valid_mask)
        backbone_hidden = model.get_temporal_hidden(regions, global_feat, valid_mask)

    print(f"输出logits shape: {logits.shape}")
    print(f"主干时序隐状态 shape: {backbone_hidden.shape}")
    print("✅ BaseVideoModel 测试通过")
