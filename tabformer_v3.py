# -*- coding: utf-8 -*-
"""
TabFormer v3: 补齐 Sparkov 配方 —— 字段级遮罩 + 个人相对金额桶 + chip。
对照: v2(整事件遮罩+chip) 窗口 0.687 / 定位 0.833, oracle 0.760。
消费流形状 → 按定稿配方本就该用字段遮罩, 这是此前的遗漏。
"""
import pickle
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

WIN = 64
N_TYPES, N_GAMT, N_CHIP = 48, 8, 3
RATIO_EDGES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_PAMT = len(RATIO_EDGES) + 2
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_BOS = len(GAP_EDGES) + 1
N_GAP = GAP_BOS + 1
FIELDS = ("type", "amt", "gap")      # amt 遮罩时 gamt+pamt 同时遮


def encode_all():
    wins = pickle.load(open("tabformer_windows.pkl", "rb"))
    tr_raw = wins["train"][:6000]
    cnt = Counter(); amts = []
    for w in tr_raw:
        cnt.update(w["mcc"].tolist()); amts.extend(w["amount"].tolist())
    vocab = {m: i for i, (m, _) in enumerate(cnt.most_common(N_TYPES - 1))}
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_GAMT + 1)[1:-1])

    def enc(w):
        med = max(float(np.median(np.abs(w["amount"]))), 1e-6)
        gaps = np.maximum(np.diff(w["ts"]), 0)
        return {"types": [vocab.get(m, N_TYPES-1) for m in w["mcc"]],
                "gamts": np.searchsorted(q, w["amount"]).tolist(),
                "pamts": (1 + np.searchsorted(RATIO_EDGES,
                                              np.abs(w["amount"]) / med)).tolist(),
                "gaps": [GAP_BOS] + np.searchsorted(GAP_EDGES, gaps,
                                                    side="left").tolist(),
                "chips": w["chip"].tolist(),
                "label": int(w["fraud"].any()), "fr": w["fraud"]}
    return ([enc(w) for w in tr_raw],
            [enc(w) for w in wins["test_w"][:1500]],
            [enc(w) for w in wins["black"][:2000]])


class FieldMEM5(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.gamt_emb = nn.Embedding(N_GAMT, 8)
        self.pamt_emb = nn.Embedding(N_PAMT, 16)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.chip_emb = nn.Embedding(N_CHIP, 8)
        self.mask_v = nn.ParameterDict({
            "m_type": nn.Parameter(torch.randn(32) * 0.02),
            "m_gamt": nn.Parameter(torch.randn(8) * 0.02),
            "m_pamt": nn.Parameter(torch.randn(16) * 0.02),
            "m_gap": nn.Parameter(torch.randn(16) * 0.02)})
        self.in_proj = nn.Linear(32 + 8 + 16 + 16 + 8, d)
        self.pos_emb = nn.Embedding(WIN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_t = nn.Linear(d, N_TYPES)
        self.head_ga = nn.Linear(d, N_GAMT)
        self.head_pa = nn.Linear(d, N_PAMT)
        self.head_g = nn.Linear(d, N_GAP)

    def head(self, f):
        return {"type": self.head_t, "gamt": self.head_ga,
                "pamt": self.head_pa, "gap": self.head_g}[f]

    def forward(self, T, mask, field):
        e = {"type": self.type_emb(T["type"]), "gamt": self.gamt_emb(T["gamt"]),
             "pamt": self.pamt_emb(T["pamt"]), "gap": self.gap_emb(T["gap"])}
        m = mask.unsqueeze(-1)
        blocked = ("gamt", "pamt") if field in ("amt", "gamt", "pamt") \
            else (field,)
        for f in blocked:
            e[f] = torch.where(m, self.mask_v["m_" + f].expand_as(e[f]), e[f])
        x = self.in_proj(torch.cat([e["type"], e["gamt"], e["pamt"],
                                    e["gap"], self.chip_emb(T["chip"])], -1))
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        h = self.encoder(x)
        tgt = "pamt" if field == "amt" else field
        return self.head(tgt)(h), tgt


def to_tensors(batch, device):
    T = {}
    for f, k in (("type", "types"), ("gamt", "gamts"), ("pamt", "pamts"),
                 ("gap", "gaps"), ("chip", "chips")):
        T[f] = torch.tensor([w[k] for w in batch], dtype=torch.long).to(device)
    return T


def train(model, wins, device, epochs=15, bs=64, lr=1e-3, ratio=0.15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.functional.cross_entropy
    model.train()
    for ep in range(epochs):
        order = list(range(len(wins)))
        random.shuffle(order)
        tot = nb = 0
        for s in range(0, len(order), bs):
            batch = [wins[j] for j in order[s:s + bs]]
            T = to_tensors(batch, device)
            field = FIELDS[nb % 3]
            mask = torch.rand(T["type"].shape) < ratio
            mask[:, 0] |= ~mask.any(1)
            mask = mask.to(device)
            logits, tgt = model(T, mask, field)
            loss = ce(logits[mask], T[tgt][mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, wins, device, stride=8, bs=128):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{f: np.zeros(WIN) for f in ("type", "pamt", "gamt", "gap")}
           for _ in wins]
    for i, w in enumerate(wins):
        out[i]["gb"] = np.array(w["gaps"]); out[i]["tp"] = np.array(w["types"])
    for field in ("type", "pamt", "gamt", "gap"):
        for r in range(stride):
            for s in range(0, len(wins), bs):
                batch = wins[s:s + bs]
                T = to_tensors(batch, device)
                pos = torch.arange(WIN)
                mask = ((pos % stride) == r)[None].expand(len(batch), WIN)
                logits, tgt = model(T, mask.to(device), field)
                idx = mask[0].nonzero().flatten()
                key = {"types": "type", "pamts": "pamt", "gamts": "gamt",
                       "gaps": "gap"}
                for i in range(len(batch)):
                    out[s+i][field][idx] = ce(logits[i, idx],
                                              T[tgt][i, idx],
                                              reduction="none").cpu().numpy()
    return out


def norm_by(pcs_tr, pcs, field, bkey, nb):
    av = np.concatenate([p[field] for p in pcs_tr])
    ab = np.concatenate([p[bkey] for p in pcs_tr])
    mu = np.full(nb, av.mean()); sd = np.full(nb, max(av.std(), 1e-3))
    for b in range(nb):
        m = ab == b
        if m.sum() >= 30:
            mu[b], sd[b] = av[m].mean(), max(av[m].std(), 1e-3)
    return [(p[field] - mu[np.clip(p[bkey], 0, nb-1)]) /
            sd[np.clip(p[bkey], 0, nb-1)] for p in pcs]


def topk_mean(a, k):
    return float(np.sort(a)[-min(k, len(a)):].mean())


def report(name, y, s):
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    print(f"  {name:<30} AUC={auc:.4f}  KS={np.max(tpr-fpr):.4f}  "
          f"R@FPR1%={r1:.1%}", flush=True)
    return auc


def main():
    device = "cpu"
    tr, tw, bl = encode_all()
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    print(f"训练 {len(tr)}, 测试 {len(tw)}白+{len(bl)}黑\n", flush=True)
    model = FieldMEM5().to(device)
    print("== FieldMEM5 训练 (字段遮罩+pamt+chip) ==", flush=True)
    train(model, tr, device)
    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, tr, device)
    pcs = per_position_ce(model, test_wins, device)
    zt = norm_by(pcs_tr, pcs, "type", "gb", N_GAP)
    zp = norm_by(pcs_tr, pcs, "pamt", "tp", N_TYPES)
    zg_ = norm_by(pcs_tr, pcs, "gamt", "tp", N_TYPES)
    combos = {"仅type": zt, "仅pamt": zp, "仅gamt": zg_,
              "type+pamt": [a+b for a, b in zip(zt, zp)],
              "type+pamt+gamt": [a+b+c for a, b, c in zip(zt, zp, zg_)]}
    best = (0, None, None)
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"top-{k} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"top-{k} {name}", z)
    print(f"\nv2(整事件遮罩+chip) = 0.687 | v3 最优 = {best[0]:.4f} ({best[1]})",
          flush=True)
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(best[2][i].tolist()); pos_y.extend(w["fr"].tolist())
    print(f"定位 AUC = {roc_auc_score(np.array(pos_y), np.array(pos_z)):.4f} "
          f"(v2 = 0.833)", flush=True)


if __name__ == "__main__":
    main()
