# -*- coding: utf-8 -*-
"""
开源数据验证 #5: BankSim (Kaggle ealaxi/banksim1)
--------------------------------------------------
4112 客户 / 59.5 万笔 / 逐笔欺诈标注 1.21%。消费流形状 → Sparkov v2 配方
(字段级遮罩)。时间只有"天"粒度(step), 间隔桶按天数分。
协议: 按客户 3300/812 切分, 训练用全部窗口(含天然污染), 测试分黑白窗口。
"""
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CSV = ("/Users/macbookpro/.cache/kagglehub/datasets/ealaxi/banksim1/"
       "versions/1/bs140513_032310.csv")
WIN = 64
N_TYPES = 16          # 15 category + OTHER
N_AMOUNT = 8
GAP_EDGES_D = [1, 2, 3, 7, 14]     # 天粒度
GAP_BOS = len(GAP_EDGES_D) + 1
N_GAP = GAP_BOS + 1
FIELDS = ("type", "amt", "gap")


def load_windows():
    df = pd.read_csv(CSV)
    for c in ("customer", "category"):
        df[c] = df[c].str.strip("'")
    df.sort_values(["customer", "step"], inplace=True, kind="mergesort")
    users = df["customer"].unique().tolist()
    rng = random.Random(SEED)
    rng.shuffle(users)
    tr_users = set(users[:3300])
    wins = {"train": [], "test_w": [], "black": []}
    for user, g in df.groupby("customer", sort=False):
        st = g["step"].to_numpy(); am = g["amount"].to_numpy()
        ca = g["category"].to_numpy(); fr = g["fraud"].to_numpy()
        for s in range(0, len(g) - WIN + 1, WIN):
            w = {"st": st[s:s+WIN], "am": am[s:s+WIN],
                 "ca": ca[s:s+WIN], "fr": fr[s:s+WIN]}
            if user in tr_users:
                wins["train"].append(w)
            elif w["fr"].any():
                wins["black"].append(w)
            else:
                wins["test_w"].append(w)
    for k in wins:
        rng.shuffle(wins[k])
    return wins


def encode_all(wins):
    tr = wins["train"][:6000]
    cnt = Counter(); amts = []
    for w in tr:
        cnt.update(w["ca"].tolist()); amts.extend(w["am"].tolist())
    vocab = {c: i for i, (c, _) in enumerate(cnt.most_common(N_TYPES - 1))}
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_AMOUNT + 1)[1:-1])

    def enc(w):
        gaps = np.maximum(np.diff(w["st"]), 0)
        return {"types": [vocab.get(c, N_TYPES-1) for c in w["ca"]],
                "amounts": np.searchsorted(q, w["am"]).tolist(),
                "gaps": [GAP_BOS] + np.searchsorted(GAP_EDGES_D, gaps,
                                                    side="left").tolist(),
                "label": int(w["fr"].any()), "fr": w["fr"]}
    return ([enc(w) for w in tr], [enc(w) for w in wins["test_w"][:1500]],
            [enc(w) for w in wins["black"]])


class FieldMEM(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.amount_emb = nn.Embedding(N_AMOUNT, 8)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.mask_t = nn.Parameter(torch.randn(32) * 0.02)
        self.mask_a = nn.Parameter(torch.randn(8) * 0.02)
        self.mask_g = nn.Parameter(torch.randn(16) * 0.02)
        self.in_proj = nn.Linear(32 + 8 + 16, d)
        self.pos_emb = nn.Embedding(WIN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_t = nn.Linear(d, N_TYPES)
        self.head_a = nn.Linear(d, N_AMOUNT)
        self.head_g = nn.Linear(d, N_GAP)

    def _head(self, f):
        return {"type": self.head_t, "amt": self.head_a, "gap": self.head_g}[f]

    def forward(self, tp, am, gp, mask, field):
        et, ea, eg = self.type_emb(tp), self.amount_emb(am), self.gap_emb(gp)
        m = mask.unsqueeze(-1)
        if field == "type":
            et = torch.where(m, self.mask_t.expand_as(et), et)
        elif field == "amt":
            ea = torch.where(m, self.mask_a.expand_as(ea), ea)
        else:
            eg = torch.where(m, self.mask_g.expand_as(eg), eg)
        x = self.in_proj(torch.cat([et, ea, eg], -1))
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        return self._head(field)(self.encoder(x))


def to_tensors(batch, device):
    tp = torch.tensor([w["types"] for w in batch], dtype=torch.long)
    am = torch.tensor([w["amounts"] for w in batch], dtype=torch.long)
    gp = torch.tensor([w["gaps"] for w in batch], dtype=torch.long)
    return tp.to(device), am.to(device), gp.to(device)


def target(f, tp, am, gp):
    return {"type": tp, "amt": am, "gap": gp}[f]


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
            tp, am, gp = to_tensors(batch, device)
            field = FIELDS[nb % 3]
            mask = torch.rand(tp.shape) < ratio
            mask[:, 0] |= ~mask.any(1)
            mask = mask.to(device)
            logits = model(tp, am, gp, mask, field)
            yv = target(field, tp, am, gp)
            loss = ce(logits[mask], yv[mask])
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
                tp, am, gp = to_tensors(batch, device)
                pos = torch.arange(WIN)
                mask = ((pos % stride) == r)[None].expand(len(batch), WIN)
                logits = model(tp, am, gp, mask.to(device), field)
                yv = target(field, tp, am, gp)
                idx = mask[0].nonzero().flatten()
                for i in range(len(batch)):
                    out[s+i][field][idx] = ce(logits[i, idx], yv[i, idx],
                                              reduction="none").cpu().numpy()
    return out


def norm_by(pcs_train, pcs, field, bkey, nb):
    all_v = np.concatenate([p[field] for p in pcs_train])
    all_b = np.concatenate([p[bkey] for p in pcs_train])
    mu = np.full(nb, all_v.mean())
    sd = np.full(nb, max(all_v.std(), 1e-3))
    for b in range(nb):
        m = all_b == b
        if m.sum() >= 30:
            mu[b], sd[b] = all_v[m].mean(), max(all_v[m].std(), 1e-3)
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
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    n_c = sum(1 for w in tr if w["label"] == 1)
    print(f"训练 {len(tr)}(含欺诈窗口 {n_c}), 测试 {len(tw)}白+{len(bl)}黑\n",
          flush=True)

    print("== 监督 oracle (5折CV) ==", flush=True)
    allw = tr + test_wins
    ya = np.array([w["label"] for w in allw])
    feats = []
    for w in allw:
        ch = np.bincount(w["types"], minlength=N_TYPES) / WIN
        feats.append(np.concatenate([ch, [np.mean(w["amounts"]),
                                          np.max(w["amounts"])]]))
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(lr_pipe, np.array(feats), ya, cv=5,
                          method="predict_proba")[:, 1]
    report("Oracle: 类别+金额+LR", ya, p)

    print("\n== FieldMEM 训练 (字段级遮罩) ==", flush=True)
    model = FieldMEM().to(device)
    train(model, tr, device)

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_field_ce(model, tr, device)
    pcs = per_position_field_ce(model, test_wins, device)
    za = norm_by(pcs_tr, pcs, "amt", "tp", N_TYPES)
    zt = norm_by(pcs_tr, pcs, "type", "gb", N_GAP)
    combos = {"仅amt|cat可见": za, "仅type": zt,
              "amt+type": [a+b for a, b in zip(za, zt)]}
    best = (0, None, None)
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"z-norm top-{k} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"top-{k} {name}", z)
    print(f"\n== 定位能力 (变体: {best[1]}) ==", flush=True)
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(best[2][i].tolist())
            pos_y.extend(w["fr"].tolist())
    pos_y = np.array(pos_y)
    print(f"  黑窗口内 {len(pos_y)} 笔, 欺诈 {int(pos_y.sum())} 笔")
    print(f"  位置级 AUC = {roc_auc_score(pos_y, np.array(pos_z)):.4f}")


if __name__ == "__main__":
    main()
