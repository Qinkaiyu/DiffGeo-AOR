import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StepQueryFusion(nn.Module):
    def __init__(self, d_feat=768, d_model=768, num_steps=4):
        super().__init__()
        # 主干映射
        self.proj_feat = nn.Linear(d_feat, d_model)

        # FiLM：把 p_{t-1} → γ、β  (调制图像/上一步特征)
        self.film = FiLMBlock(d_model)

        # Step / Position embedding（固定）
        self.step_tab = nn.Embedding(num_steps, d_model)
        # for p in self.step_tab.parameters():
        #     p.requires_grad = False            # 冻结

        # 单头或多头都行，保持 cross-attn 结构
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)

    def forward(self, feat, p_prev, step_t, return_intermediate: bool = False):
        """
        feat   : (B*, d_feat)  t=0 时是图像 CLS；t>0 时 cond_{t-1}
        p_prev : (B*, 1)       上一步 sigmoid 概率
        step_t : int 0 … 3     当前自回归步
        """
        if feat.dim() == 2:
            feat = feat.unsqueeze(1)           # (B*,1,d_feat)
        # feat = feat.repeat(diffusion_batch_mul, 1)
        # ① FiLM 注入 p_{t-1}
        F = self.proj_feat(feat)               # (B*,1,d)
        F_mod = self.film(F, p_prev)           # (B*,1,d)
        # ② 仅用 step 位置向量做 Query
        step_q = self.step_tab.weight[step_t]  # (d)
        q = step_q.unsqueeze(0).expand(F_mod.size(0), -1).unsqueeze(1)  # (B*,1,d)
        # ③ Cross-Attention (Q=step, K/V=F_mod)
        cond, _ = self.attn(q, F_mod, F_mod)   # (B*,1,d) → squeeze
        # print('cond',cond.shape)
        # print('feat',feat.shape)
        cond = cond.squeeze(1)+feat.squeeze(1)
        if return_intermediate:
            return cond, {
                "film_before": F,
                "film_after": F_mod,
                "p_prev": p_prev,
                "step_t": int(step_t),
            }
        return cond                 # (B*, d)
class FiLMBlock(nn.Module):
    """
    p_prev → γ,β  →  F_mod = γ·F + β
    """
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )

    def forward(self, F, p_prev):
        """

        """
        gamma_beta = self.mlp(p_prev)            # (B, 2d)
        γ, β = gamma_beta.chunk(2, dim=-1)       # (B, d) ×2
        γ = γ.unsqueeze(1)                       # (B,1,d)
        β = β.unsqueeze(1)
        return γ * F + β        
class TopKAttnPool(nn.Module):
    """
    简易 top-k attention pooling：
    - 用 1×1 conv 产生每个 patch 的打分 score
    - softmax 后取权重最大的 k 个 patch（其余权重自动趋近 0）
    """
    def __init__(self, d_model=768, k=20):
        super().__init__()
        self.k = k
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)          # (B,N,1)
        )

    def forward(self, x):
        cls, patches = x[:, :1, :], x[:, 1:, :]          # B×1×d ,  B×N×d
        score = self.scorer(patches).squeeze(-1)         # B×N
        # softmax 前先把除 top-k 之外的位置置 -inf
        topk_val, topk_idx = torch.topk(score, self.k, dim=1)
        mask = torch.full_like(score, -float('inf'))
        mask.scatter_(1, topk_idx, 0.)
        attn = torch.softmax(score + mask, dim=1)        # B×N
        pooled = (patches * attn.unsqueeze(-1)).sum(1)   # B×d
        return pooled
class GeMPool(nn.Module):
    """
    Generalized-Mean Pooling.
    p=1  -> GAP
    p→∞ -> max-pool (soft-max 极限)
    """
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x):
        # 取所有 patch，不包含 CLS
        patches = x[:, 1:, :]                       # B×N×d
        p = torch.clamp(self.p, min=1.)             # 约束 p ≥ 1
        gem = patches.clamp(min=self.eps).pow(p)
        gem = gem.mean(dim=1).pow(1. / p)           # B×d
        return gem
        
class RankReference(nn.Module):
    def __init__(self, num_ranks: int = 4,
                 #cls+gap d_model=1536
                 #d = cat[768,768]， vit
                 #d = 512, resnet3d
                 d_model: int = 512,
                 scale: float = 20.,     # cos → logits 的放大因子
                 margin: float = 0.3,    # 距离损失的 margin
                 tau: int = 0):          # 跳过几档再算 dist_neg
        super().__init__()
        self.R = nn.Parameter(torch.randn(num_ranks, d_model))
        nn.init.kaiming_uniform_(self.R, a=math.sqrt(7))
        self.scale  = scale
        self.margin = margin
        self.tau    = tau

    # ----------------------------------------------
    # cond : (B, d)  -- 本步融合后的特征
    # labels : (B,)  -- 0…C-1 实际类别
    # ----------------------------------------------
    def order_metric_gol(self, cond: torch.Tensor,
                               labels: torch.Tensor,
                               return_detail: bool = False):
        B, d = cond.shape
        
        C     = self.R.size(0)
        # ---------- 1. 自适应 margin (类内方差版本) ----------
        beta = 0.4          # 缩放系数  (建议 0.3~0.5)
        ema  = 0.9          # EMA 平滑 (历史占 90%)
        with torch.no_grad():
            # 计算每个类别当前 batch 的 “类内标准差” σ_c
            sigmas = []
            for c in range(C):
                mask = labels == c                       # 该类是否在 batch 中
                if mask.any():
                    # cond[mask] : N_c × d
                    sigmas.append(cond[mask].std(0).mean())  # σ_c
                else:
                    # 该类没样本，用历史 R 粗略估计
                    sigmas.append((self.R[c] - self.R.mean(0)).norm()
                                / math.sqrt(d))
            sigma_mean = torch.stack(sigmas).mean()          # σ̄

            # 新 margin（缩放 β，再做 EMA）
            m_new = beta * sigma_mean
            self.margin = ema * self.margin + (1 - ema) * m_new
        # ---------- 归一化 ----------
        cond_n = F.normalize(cond, dim=-1)           # (B,d)
        R_n    = F.normalize(self.R,  dim=-1)        # (C,d)

        # ---------- direction (前/后) ----------
        idx_pos  = labels
        idx_prev = (labels-1).clamp(0, C-1)
        idx_next = (labels+1).clamp(0, C-1)

        dir_fwd  = F.normalize(R_n[idx_next] - R_n[idx_pos], dim=-1)  # (B,d)
        dir_back = F.normalize(R_n[idx_pos]  - R_n[idx_prev], dim=-1) # (B,d)

        # 2-class logits: [backward , forward]
        logit_back = (cond_n * dir_back).sum(-1) * self.scale
        logit_fwd  = (cond_n * dir_fwd ).sum(-1) * self.scale
        logits     = torch.stack([logit_back, logit_fwd], dim=1)      # (B,2)

        # GT：除去边界，全部应该“朝前”(class = 1)
        gt = torch.ones(B, device=labels.device, dtype=torch.long)
        gt[labels == 0]     = 0    # 最低档只能往后
        gt[labels == C-1]   = 0    # 最高档无 forward，给 0 但可忽略
        L_ord_each = F.cross_entropy(logits, gt, reduction="none")
        L_ord = L_ord_each.mean()

        # ---------- metric 距离 ----------
        idx_far = (labels + 1 + self.tau).clamp(0, C-1)   # 跳过 τ 档
        dist_pos = (cond_n - R_n[idx_pos]).norm(dim=1)
        dist_neg = (cond_n - R_n[idx_far]).norm(dim=1)
        margin_violation = self.margin + dist_pos - dist_neg
        L_met_each = F.relu(margin_violation)
        L_met = L_met_each.mean()

        if return_detail:
            margin_scalar = float(self.margin.item()) if isinstance(self.margin, torch.Tensor) else float(self.margin)
            return L_ord, L_met, {
                "ord_each": L_ord_each,
                "met_each": L_met_each,
                "logit_back": logit_back,
                "logit_fwd": logit_fwd,
                "ord_target": gt,
                "ord_gap": (logit_fwd - logit_back),
                "dist_pos": dist_pos,
                "dist_neg": dist_neg,
                "margin_violation": margin_violation,
                "margin_scalar": margin_scalar,
            }
        return L_ord, L_met
class RankEmbed(nn.Module):

    def __init__(self, num_ranks: int, d: int,mlp_hidden: int = None):
        super().__init__()
        self.table = nn.Embedding(num_ranks, d)
        self.mlp = nn.Sequential(
                nn.Linear(d, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, d))


    def forward(self, k: torch.LongTensor):  # shape = (B,)
        e = self.table(k)                    # (B, d)
        return self.mlp(e)                   # (B, d)
class SimpleFusion(nn.Module):
    def __init__(self, feature_dim=768, context_dim=1, emb_dim=768):
        super(SimpleFusion, self).__init__()
        self.position_embedding = nn.Embedding(4, emb_dim)

        #affine fusion
        self.proj_feature = nn.Linear(feature_dim, emb_dim)
        self.proj_context = nn.Embedding(5, emb_dim)
        # cross-attn
        # self.proj_context = nn.Linear(context_dim, emb_dim)
        # self.proj_context = nn.Embedding(5, emb_dim)
    def forward(self, features, context,t):
        '''
        :param features: [batch_size, 1024]
        :param context: [batch_size, 5]
        :return: [batch_size, 1024]
        '''
        # 投影到嵌入维度
        # affine fusion
        # features = self.proj_feature(features)  # [batch_size, emb_dim]\
        # context = self.proj_context(context.long())  # [batch_size, emb_dim]
        # context = context.squeeze(1)
        # position = torch.tensor(t).repeat(context.shape[0],1)
        # position = position.squeeze(1).to(device='cuda')
        # position_embedding = self.position_embedding(position)
        # context = context + position_embedding
        # out = context * features + features  # [batch_size, emb_dim]
        # cross-attn 
        features = self.proj_feature(features)  # [batch_size, emb_dim]
        context = self.proj_context(context.long())  # [batch_size, emb_dim]
        position = torch.tensor(t).repeat(context.shape[0],1).to(device='cuda')
        position_embedding = self.position_embedding(position)
        position_embedding = position_embedding.squeeze(1)
        context = context.squeeze(1)
        context = context + position_embedding
        attention_score = torch.matmul(features,context.transpose(0,1))
        attention_weight = F.softmax(attention_score,dim=1)
        attention_context = torch.matmul(attention_weight,context)
        out = attention_context + features  
        return out

def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)
    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0),
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m / torch.max(m)
    return imp
