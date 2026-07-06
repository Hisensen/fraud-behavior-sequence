# -*- coding: utf-8 -*-
"""
Sparkov v3: 在 v2(字段级遮罩, AUC 0.848)基础上加"个人相对金额桶"字段。
pamt = 金额 / 窗口中位金额, 按 [0.25,0.5,1,2,4,8] 分桶(0=N/A)。
问题: 真实卡数据上, "相对个人基线"的金额信息比全局分桶多赚多少?
对照: 同协议同种子, 仅编码差一个字段(v2 三字段 vs v3 四字段)。
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from sparkov_experiment import (load_windows, N_TYPES, N_AMOUNT, N_GAP,
                                GAP_EDGES, GAP_BOS, WIN, SEED)
from collections import Counter

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RATIO_EDGES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_PAMT = len(RATIO_EDGES) + 2
FIELDS = ("type", "amt", "gap", "pamt")


def encode_all(wins):
    tr_raw = wins["train"][:6000]
    cnt = Counter(); amts = []
    for w in tr_raw:
        cnt.update(w["cat"].tolist()); amts.extend(w["amt"].tolist())
    vocab = {c: i for i, (c, _) in enumerate(cnt.most_common(N_TYPES - 1))}
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_AMOUNT + 1)[1:-1])

    def enc(w):
        med = max(float(np.median(w["amt"])), 1e-6)
        gaps = np.maximum(np.diff(w["ts"]), 0)
        return {"types": [vocab.get(c, N_TYPES-1) for c in w["cat"]],
                "amounts": np.searchsorted(q, w["amt"]).tolist(),
                "pamts": (1 + np.searchsorted(RATIO_EDGES,
                                              w["amt"] / med)).tolist(),
                "gaps": [GAP_BOS] + np.searchsorted(GAP_EDGES, gaps,
                                                    side="left").tolist(),
                "label": int(w["fr"].any()), "fr": w["fr"]}
    return ([enc(w) for w in tr_raw], [enc(w) for w in wins["test_w"][:1500]],
            [enc(w) for w in wins["black"]])


class FieldMEM4(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.amount_emb = nn.Embedding(N_AMOUNT, 8)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.pamt_emb = nn.Embedding(N_PAMT, 16)
        self.mask_v = nn.ParameterDict({
            "m_type": nn.Parameter(torch.randn(32) * 0.02),
            "m_amt": nn.Parameter(torch.randn(8) * 0.02),
            "m_gap": nn.Parameter(torch.randn(16) * 0.02),
            "m_pamt": nn.Parameter(torch.randn(16) * 0.02)})
        self.in_proj = nn.Linear(32 + 8 + 16 + 16, d)
        self.pos_emb = nn.Embedding(WIN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.heads = nn.ModuleDict({"h_type": nn.Linear(d, N_TYPES),
                                    "h_amt": nn.Linear(d, N_AMOUNT),
                                    "h_gap": nn.Linear(d, N_GAP),
                                    "h_pamt": nn.Linear(d, N_PAMT)})

    def forward(self, T, mask, field):
        e = {"type": self.type_emb(T["type"]), "amt": self.amount_emb(T["amt"]),
             "gap": self.gap_emb(T["gap"]), "pamt": self.pamt_emb(T["pamt"])}
        m = mask.unsqueeze(-1)
        # 遮 amt 时同时遮 pamt(两者都编码金额, 只遮一个会泄漏答案), 反之亦然
        blocked = {"amt": ("amt", "pamt"), "pamt": ("amt", "pamt")}.get(
            field, (field,))
        for f in blocked:
            e[f] = torch.where(m, self.mask_v["m_" + f].expand_as(e[f]), e[f])
        x = self.in_proj(torch.cat([e["type"], e["amt"], e["gap"],
                                    e["pamt"]], -1))
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        return self.heads["h_" + field](self.encoder(x))


def to_tensors(batch, device):
    T = {}
    for f, key in (("type", "types"), ("amt", "amounts"),
                   ("gap", "gaps"), ("pamt", "pamts")):
        T[f] = torch.tensor([w[key] for w in batch], dtype=torch.long).to(device)
    return T


def train(model, wins, device, epochs=20, bs=64, lr=1e-3, ratio=0.15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.functional.cross_entropy
    model.train()
    for ep in range(epochs):
        order = list(range(len(wins)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [wins[j] for j in order[s:s + bs]]
            T = to_tensors(batch, device)
            field = FIELDS[nb % 4]
            mask = torch.rand(T["type"].shape) < ratio
            mask[:, 0] |= ~mask.any(1)
            mask = mask.to(device)
            logits = model(T, mask, field)
            loss = ce(logits[mask], T[field][mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_field_ce(model, wins, device, stride=8, bs=128):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{f: np.zeros(WIN) for f in FIELDS} for _ in wins]
    for i, w in enumerate(wins):
        out[i]["gb"] = np.array(w["gaps"])
        out[i]["tp"] = np.array(w["types"])
    for field in FIELDS:
        for r in range(stride):
            for s in range(0, len(wins), bs):
                batch = wins[s:s + bs]
                T = to_tensors(batch, device)
                pos = torch.arange(WIN)
                mask = ((pos % stride) == r)[None].expand(len(batch), WIN)
                logits = model(T, mask.to(device), field)
                idx = mask[0].nonzero().flatten()
                for i in range(len(batch)):
                    out[s+i][field][idx] = ce(logits[i, idx],
                                              T[field][i, idx],
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
    print(f"  {name:<32} AUC={auc:.4f}  KS={np.max(tpr-fpr):.4f}  "
          f"R@FPR1%={r1:.1%}", flush=True)
    return auc


def main():
    device = "cpu"
    wins = load_windows()
    tr, tw, bl = encode_all(wins)
    tr = tr[:6000]
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    print(f"训练 {len(tr)}, 测试 {len(tw)}白+{len(bl)}黑\n", flush=True)

    print("== FieldMEM4 训练 (4字段轮换遮罩, 含个人相对金额) ==", flush=True)
    model = FieldMEM4().to(device)
    train(model, tr, device)

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_field_ce(model, tr, device)
    pcs = per_position_field_ce(model, test_wins, device)
    za = norm_by(pcs_tr, pcs, "amt", "tp", N_TYPES)
    zp = norm_by(pcs_tr, pcs, "pamt", "tp", N_TYPES)
    zt = norm_by(pcs_tr, pcs, "type", "gb", N_GAP)
    combos = {"仅type": zt, "仅amt(全局)": za, "仅pamt(个人)": zp,
              "type+pamt": [a+b for a, b in zip(zt, zp)],
              "type+amt+pamt": [a+b+c for a, b, c in zip(zt, za, zp)]}
    best = (0, None, None)
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"z-norm top-{k} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"top-{k} {name}", z)
    print(f"\nv2 基线(三字段) = 0.848 | v3 最优 = {best[0]:.4f} ({best[1]})",
          flush=True)
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(best[2][i].tolist())
            pos_y.extend(w["fr"].tolist())
    print(f"定位 AUC = {roc_auc_score(np.array(pos_y), np.array(pos_z)):.4f} "
          f"(v2 = 0.696)", flush=True)


if __name__ == "__main__":
    main()
