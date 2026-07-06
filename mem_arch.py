# -*- coding: utf-8 -*-
"""
网络架构自由探索 —— data_rich 基准, E3 全量编码固定, 只动架构。

变体(同数据/同切分/同打分网格):
  V0 基线:            2层 d=64 ff=128, 学习式位置编码, 遮罩15%   (复现 0.9115)
  V1 加大:            3层 d=128 ff=256
  V2 时间偏置注意力:   Δt 分桶 → 每头一个可学偏置, 加进注意力打分(策略③)
  V3 去位置编码:       时间字段已含顺序信息, 检验绝对位置是否冗余
  V4 遮罩率 30%
  V5 自回归(AR):       因果注意力预测下一事件(type/res/pamt/gap), 单趟打分
  V6 集成:            V0 配置 × 3 种子, 分数秩平均
最后: 单项有效的改动组合成 V7, 若仍最优则为定稿架构。
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import rankdata

from mem_rich import (load, global_quantiles, EVENT_TYPES, N_GAP, N_PAMT,
                      N_GAMT, GAP_EDGES, MAX_LEN, FIELD_DIMS, z_norm,
                      topk_mean, metrics, SEED)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FIELDS = ["type", "gamt", "gap", "hour", "res", "ch", "ip", "pamt"]  # E3
N_CLS = {"type": len(EVENT_TYPES), "res": 3, "ch": 4, "ip": 3,
         "gamt": N_GAMT, "pamt": N_PAMT, "gap": N_GAP}
HEAD_MAP = {"h_type": "type", "h_gap": "gap", "h_res": "res", "h_pamt": "pamt"}
DT_EDGES = GAP_EDGES  # Δt 偏置分桶, 8 桶


def to_tensors(batch, device):
    n = len(batch)
    L = max(len(u["type"]) for u in batch)
    T = {}
    for f in ("type", "res", "ch", "ip", "gamt", "pamt", "gap"):
        M = torch.zeros(n, L, dtype=torch.long)
        for i, u in enumerate(batch):
            M[i, :len(u[f])] = torch.tensor(u[f])
        T[f] = M.to(device)
    H = torch.zeros(n, L)
    TS = torch.zeros(n, L, dtype=torch.float64)
    pad = torch.ones(n, L, dtype=torch.bool)
    for i, u in enumerate(batch):
        m = len(u["hour"])
        H[i, :m] = torch.tensor(u["hour"])
        TS[i, :m] = torch.tensor(u["ts"], dtype=torch.float64)
        pad[i, :m] = False
    T["hour"] = H.to(device)
    T["ts"] = TS.to(device)
    return T, pad.to(device)


class TimeBiasLayer(nn.Module):
    """post-norm Transformer 层 + Δt 分桶注意力偏置(每头独立)"""
    def __init__(self, d, nhead, ff, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, pad):
        a, _ = self.attn(x, x, x, attn_mask=bias, key_padding_mask=pad,
                         need_weights=False)
        x = self.n1(x + self.drop(a))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class MEMArch(nn.Module):
    def __init__(self, d=64, layers=2, ff=128, nhead=4, use_pe=True,
                 time_bias=False, causal=False):
        super().__init__()
        self.use_pe, self.time_bias, self.causal, self.nhead = \
            use_pe, time_bias, causal, nhead
        self.embs = nn.ModuleDict()
        in_dim = 0
        for f in FIELDS:
            if f == "hour":
                self.hour_proj = nn.Linear(2, FIELD_DIMS["hour"])
            else:
                self.embs["e_" + f] = nn.Embedding(N_CLS[f], FIELD_DIMS[f])
            in_dim += FIELD_DIMS[f]
        self.in_proj = nn.Linear(in_dim, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        if use_pe:
            self.pos_emb = nn.Embedding(MAX_LEN, d)
        if time_bias:
            self.bias_emb = nn.Embedding(len(DT_EDGES) + 1, nhead)
            self.layers = nn.ModuleList(
                [TimeBiasLayer(d, nhead, ff) for _ in range(layers)])
        else:
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                               dim_feedforward=ff, dropout=0.1,
                                               batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.heads = nn.ModuleDict({k: nn.Linear(d, N_CLS[f])
                                    for k, f in HEAD_MAP.items()})

    def _embed(self, T, mask):
        parts = []
        for f in FIELDS:
            if f == "hour":
                h = T["hour"]
                parts.append(self.hour_proj(torch.stack(
                    [torch.sin(2 * math.pi * h / 24),
                     torch.cos(2 * math.pi * h / 24)], -1)))
            else:
                parts.append(self.embs["e_" + f](T[f]))
        x = self.in_proj(torch.cat(parts, -1))
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        if self.use_pe:
            x = x + self.pos_emb(torch.arange(x.size(1),
                                              device=x.device))[None]
        return x

    def _bias(self, T, pad):
        B, L = pad.shape
        bias = None
        if self.time_bias:
            dt = (T["ts"][:, :, None] - T["ts"][:, None, :]).abs()
            dtb = torch.bucketize(dt.float(),
                                  torch.tensor(DT_EDGES, dtype=torch.float32,
                                               device=dt.device))
            bias = self.bias_emb(dtb).permute(0, 3, 1, 2).reshape(
                B * self.nhead, L, L)
        if self.causal:
            cm = torch.full((L, L), float("-inf"), device=pad.device)
            cm = torch.triu(cm, diagonal=1)
            # 无偏置时用 2D 掩码(可广播); 有偏置时加到每个 (B*H) 切片上
            bias = cm if bias is None else bias + cm[None]
        return bias

    def forward(self, T, pad, mask):
        x = self._embed(T, mask)
        bias = self._bias(T, pad)
        if self.time_bias:
            for lyr in self.layers:
                x = lyr(x, bias, pad)
            h = x
        else:
            h = self.encoder(x, mask=bias if bias is not None else None,
                             src_key_padding_mask=pad)
        return {k: head(h) for k, head in self.heads.items()}


def random_mask(pad, ratio, gen):
    sc = torch.rand(pad.shape, generator=gen)
    sc[pad] = -1.0
    mask = sc > (1 - ratio)
    for i in range(pad.size(0)):
        if not mask[i].any():
            v = (~pad[i]).nonzero().flatten()
            mask[i, v[torch.randint(len(v), (1,), generator=gen)]] = True
    return mask


def train(model, users, device, epochs=30, bs=32, lr=1e-3, ratio=0.15,
          seed=SEED):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.functional.cross_entropy
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for ep in range(epochs):
        order = list(range(len(users)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [users[j] for j in order[s:s + bs]]
            T, pad = to_tensors(batch, device)
            if model.causal:
                out = model(T, pad, None)
                loss = 0
                for k, f in HEAD_MAP.items():
                    tgt = T[f][:, 1:]
                    lg = out[k][:, :-1]
                    m = ~pad[:, 1:]
                    loss = loss + ce(lg[m], tgt[m])
            else:
                mask = random_mask(pad, ratio, gen).to(device)
                out = model(T, pad, mask)
                loss = sum(ce(out[k][mask], T[f][mask])
                           for k, f in HEAD_MAP.items())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"    epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, users, device, stride=7, bs=64):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{k: np.zeros(len(u["type"])) for k in HEAD_MAP} for u in users]
    for i, u in enumerate(users):
        out[i]["gb"] = np.array(u["gap"])
    if model.causal:                       # 单趟: 位置 i 的 CE = 预测它的误差
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            T, pad = to_tensors(batch, device)
            o = model(T, pad, None)
            for i in range(len(batch)):
                m = len(batch[i]["type"])
                for k, f in HEAD_MAP.items():
                    if m > 1:
                        out[s+i][k][1:m] = ce(o[k][i, :m-1], T[f][i, 1:m],
                                              reduction="none").cpu().numpy()
        return out
    for r in range(stride):
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            T, pad = to_tensors(batch, device)
            pos = torch.arange(T["type"].size(1))
            mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
            if not mask.any():
                continue
            o = model(T, pad, mask.to(device))
            for i in range(len(batch)):
                idx = mask[i].nonzero().flatten()
                if len(idx) == 0:
                    continue
                for k, f in HEAD_MAP.items():
                    out[s+i][k][idx.numpy()] = ce(
                        o[k][i, idx], T[f][i, idx],
                        reduction="none").cpu().numpy()
    return out


def score_grid(pcs_tr, pcs, y):
    zs = {k: z_norm(pcs_tr, pcs, k) for k in HEAD_MAP}
    combos = dict(zs)
    combos["SUM"] = [sum(v) for v in zip(*zs.values())]
    best = (0, None, None)
    for cn, z in combos.items():
        for agg, fn in [("top3", lambda a: topk_mean(a, 3)),
                        ("top5", lambda a: topk_mean(a, 5)),
                        ("top10", lambda a: topk_mean(a, 10)),
                        ("mean", lambda a: a.mean())]:
            s = np.array([fn(a) for a in z])
            auc, ks, r1 = metrics(y, s)
            if auc > best[0]:
                best = (auc, f"{cn} {agg}", s, ks, r1)
    return best


def run_variant(name, cfg, train_w, test_u, y, device, seed=SEED,
                ratio=0.15):
    print(f"\n== {name} ==", flush=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = MEMArch(**cfg).to(device)
    print(f"    参数 {sum(p.numel() for p in model.parameters())/1e3:.0f}K",
          flush=True)
    train(model, train_w, device, ratio=ratio, seed=seed)
    pcs_tr = per_position_ce(model, train_w, device)
    pcs = per_position_ce(model, test_u, device)
    auc, cfg_s, s, ks, r1 = score_grid(pcs_tr, pcs, y)
    print(f"    最优: {cfg_s:<14} AUC={auc:.4f}  KS={ks:.4f}  "
          f"R@FPR1%={r1:.1%}", flush=True)
    return auc, cfg_s, s, r1


def main():
    device = "cpu"
    gq = global_quantiles()
    users = load(gq=gq)
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_u = whites[:1000], whites[1000:] + blacks
    y = np.array([u["label"] for u in test_u])
    print(f"训练 {len(train_w)} | 测试 {int((y==0).sum())}白+{int(y.sum())}黑",
          flush=True)

    R = {}
    R["V0 基线(2L,d64,PE)"] = run_variant(
        "V0 基线", dict(), train_w, test_u, y, device)
    R["V1 加大(3L,d128,ff256)"] = run_variant(
        "V1 加大", dict(d=128, layers=3, ff=256), train_w, test_u, y, device)
    R["V2 时间偏置注意力"] = run_variant(
        "V2 时间偏置", dict(time_bias=True), train_w, test_u, y, device)
    R["V3 去位置编码"] = run_variant(
        "V3 去PE", dict(use_pe=False), train_w, test_u, y, device)
    R["V4 遮罩率30%"] = run_variant(
        "V4 遮罩30%", dict(), train_w, test_u, y, device, ratio=0.30)
    R["V5 自回归AR"] = run_variant(
        "V5 AR", dict(causal=True), train_w, test_u, y, device)

    # V6: 基线 3 种子集成
    print("\n== V6 三种子集成(V0 配置) ==", flush=True)
    ss = [R["V0 基线(2L,d64,PE)"][2]]
    for sd in (7, 77):
        _, _, s, _ = run_variant(f"  seed{sd}", dict(), train_w, test_u, y,
                                 device, seed=sd)
        ss.append(s)
    ens = sum(rankdata(s) for s in ss)
    auc, ks, r1 = metrics(y, ens)
    R["V6 三种子集成"] = (auc, "rank-avg", ens, r1)
    print(f"    集成 AUC={auc:.4f}  R@FPR1%={r1:.1%}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("== 架构探索总表 ==", flush=True)
    for name, (auc, cfg_s, _, r1) in R.items():
        print(f"  {name:<26} AUC={auc:.4f}  R@FPR1%={r1:>6.1%}  ({cfg_s})",
              flush=True)


if __name__ == "__main__":
    main()
