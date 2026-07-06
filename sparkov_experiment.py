# -*- coding: utf-8 -*-
"""
真实数据验证 #2: Sparkov 信用卡欺诈 (Kaggle kartik2112/fraud-detection)
------------------------------------------------------------------
999 客户 / 185 万笔 / 欺诈率 0.52%, 逐笔 is_fraud 标注。
协议(完全无监督, 按客户防泄漏):
  - 800 客户的全部窗口直接训练(含 ~0.5% 天然欺诈污染, 不筛)
  - 199 个陌生客户测试: 含欺诈窗口=黑, 无欺诈窗口=白
模型: 4 字段编码(类别/金额档/间隔桶/时刻) + 3 预测头(类别+间隔+金额)。
卡欺诈签名主要在金额×类别×时段, 金额头是本实验新增。
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA = "/Users/macbookpro/.cache/kagglehub/datasets/kartik2112/fraud-detection/versions/1/"
WIN = 64
N_TYPES = 15   # 14 类别 + OTHER
N_AMOUNT = 8
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_BOS = len(GAP_EDGES) + 1
N_GAP = GAP_BOS + 1
N_TRAIN_WIN = 6000
N_TESTW_WIN = 1500


def load_windows():
    cols = ["cc_num", "category", "amt", "unix_time", "is_fraud"]
    df = pd.concat([pd.read_csv(DATA + "fraudTrain.csv", usecols=cols),
                    pd.read_csv(DATA + "fraudTest.csv", usecols=cols)],
                   ignore_index=True)
    df.sort_values(["cc_num", "unix_time"], inplace=True, kind="mergesort")
    users = df["cc_num"].unique().tolist()
    rng = random.Random(SEED)
    rng.shuffle(users)
    tr_users = set(users[:800])

    wins = {"train": [], "test_w": [], "black": []}
    for user, g in df.groupby("cc_num", sort=False):
        ts = g["unix_time"].to_numpy()
        amt = g["amt"].to_numpy()
        cat = g["category"].to_numpy()
        fr = g["is_fraud"].to_numpy()
        for s in range(0, len(g) - WIN + 1, WIN):
            w = {"ts": ts[s:s+WIN], "amt": amt[s:s+WIN],
                 "cat": cat[s:s+WIN], "fr": fr[s:s+WIN], "user": user}
            if user in tr_users:
                wins["train"].append(w)          # 不筛欺诈 → 天然污染
            elif w["fr"].any():
                wins["black"].append(w)
            else:
                wins["test_w"].append(w)
    for k in wins:
        rng.shuffle(wins[k])
    return wins


def encode_all(wins):
    tr = wins["train"][:N_TRAIN_WIN]
    cat_cnt = Counter()
    amts = []
    for w in tr:
        cat_cnt.update(w["cat"].tolist())
        amts.extend(w["amt"].tolist())
    vocab = {c: i for i, (c, _) in enumerate(cat_cnt.most_common(N_TYPES - 1))}
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_AMOUNT + 1)[1:-1])

    def enc(w):
        gaps = np.maximum(np.diff(w["ts"]), 0)
        return {"types": [vocab.get(c, N_TYPES - 1) for c in w["cat"]],
                "amounts": np.searchsorted(q, w["amt"]).tolist(),
                "gaps": [GAP_BOS] + np.searchsorted(GAP_EDGES, gaps,
                                                    side="left").tolist(),
                "hours": ((w["ts"] % 86400) / 3600.0).tolist(),
                "label": int(w["fr"].any()), "fr": w["fr"], "user": w["user"]}

    return ([enc(w) for w in tr],
            [enc(w) for w in wins["test_w"][:N_TESTW_WIN]],
            [enc(w) for w in wins["black"]])


class MEM4(nn.Module):
    """4 字段输入, 3 预测头(type + gap + amount)"""
    def __init__(self, d=64):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.amount_emb = nn.Embedding(N_AMOUNT, 8)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.hour_proj = nn.Linear(2, 8)
        self.in_proj = nn.Linear(32 + 8 + 16 + 8, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.pos_emb = nn.Embedding(WIN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_type = nn.Linear(d, N_TYPES)
        self.head_gap = nn.Linear(d, N_GAP)
        self.head_amt = nn.Linear(d, N_AMOUNT)

    def forward(self, tp, am, gp, hr, mask):
        x = self.in_proj(torch.cat([self.type_emb(tp), self.amount_emb(am),
                                    self.gap_emb(gp), self.hour_proj(hr)], -1))
        x = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        h = self.encoder(x)
        return self.head_type(h), self.head_gap(h), self.head_amt(h)


def to_tensors(batch, device):
    tp = torch.tensor([w["types"] for w in batch], dtype=torch.long)
    am = torch.tensor([w["amounts"] for w in batch], dtype=torch.long)
    gp = torch.tensor([w["gaps"] for w in batch], dtype=torch.long)
    h = torch.tensor([w["hours"] for w in batch])
    hr = torch.stack([torch.sin(2 * math.pi * h / 24),
                      torch.cos(2 * math.pi * h / 24)], -1)
    return tp.to(device), am.to(device), gp.to(device), hr.to(device)


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
            tp, am, gp, hr = to_tensors(batch, device)
            mask = torch.rand(tp.shape) < ratio
            mask[:, 0] |= ~mask.any(1)
            mask = mask.to(device)
            lt, lg, la = model(tp, am, gp, hr, mask)
            loss = (ce(lt[mask], tp[mask]) + ce(lg[mask], gp[mask]) +
                    ce(la[mask], am[mask]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, wins, device, stride=8, bs=128):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{"t": np.zeros(WIN), "g": np.zeros(WIN), "a": np.zeros(WIN),
            "gb": np.array(w["gaps"])} for w in wins]
    for r in range(stride):
        for s in range(0, len(wins), bs):
            batch = wins[s:s + bs]
            tp, am, gp, hr = to_tensors(batch, device)
            pos = torch.arange(WIN)
            mask = ((pos % stride) == r)[None].expand(len(batch), WIN)
            lt, lg, la = model(tp, am, gp, hr, mask.to(device))
            idx = mask[0].nonzero().flatten()
            for i in range(len(batch)):
                out[s + i]["t"][idx] = ce(lt[i, idx], tp[i, idx],
                                          reduction="none").cpu().numpy()
                out[s + i]["g"][idx] = ce(lg[i, idx], gp[i, idx],
                                          reduction="none").cpu().numpy()
                out[s + i]["a"][idx] = ce(la[i, idx], am[i, idx],
                                          reduction="none").cpu().numpy()
    return out


def bucket_norm(pcs_train, pcs, key):
    all_v = np.concatenate([p[key] for p in pcs_train])
    all_b = np.concatenate([p["gb"] for p in pcs_train])
    mu = np.full(N_GAP, all_v.mean())
    sd = np.full(N_GAP, max(all_v.std(), 1e-3))
    for b in range(N_GAP):
        m = all_b == b
        if m.sum() >= 30:
            mu[b], sd[b] = all_v[m].mean(), max(all_v[m].std(), 1e-3)
    return [(p[key] - mu[np.clip(p["gb"], 0, N_GAP-1)]) /
            sd[np.clip(p["gb"], 0, N_GAP-1)] for p in pcs]


def topk_mean(a, k):
    return float(np.sort(a)[-min(k, len(a)):].mean())


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return (roc_auc_score(y, s), float(np.max(tpr - fpr)),
            float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]))


def report(name, y, s):
    auc, ks, r1 = metrics(y, s)
    print(f"  {name:<34} AUC={auc:.4f}  KS={ks:.4f}  R@FPR1%={r1:.1%}",
          flush=True)
    return auc


def main():
    device = "cpu"
    print("== 加载 Sparkov 并切窗口 ==", flush=True)
    wins = load_windows()
    print(f"窗口池: 训练 {len(wins['train'])}(含天然污染), "
          f"测试白 {len(wins['test_w'])}, 黑 {len(wins['black'])}", flush=True)
    tr, tw, bl = encode_all(wins)
    n_contam = sum(1 for w in tr if w["label"] == 1)
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    print(f"编码完成: 训练 {len(tr)}(其中含欺诈窗口 {n_contam} 个 = "
          f"{n_contam/len(tr):.1%}), 测试 {len(tw)}白+{len(bl)}黑\n", flush=True)

    # ---- 监督 oracle(用标签, 5折CV) ----
    print("== 监督 oracle 基线 ==", flush=True)
    allw = tr + test_wins
    ya = np.array([w["label"] for w in allw])
    feats = []
    for w in allw:
        cath = np.bincount(w["types"], minlength=N_TYPES) / WIN
        feats.append(np.concatenate([
            cath,
            [np.mean(w["amounts"]), np.max(w["amounts"]),
             np.mean(np.array(w["gaps"]) <= 2),
             np.mean((np.array(w["hours"]) < 6) | (np.array(w["hours"]) > 22)),
             np.mean(w["hours"])]]))
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(lr_pipe, np.array(feats), ya, cv=5,
                          method="predict_proba")[:, 1]
    report("Oracle: 类别+金额+时段统计+LR", ya, p)

    # ---- MEM4 ----
    print("\n== MEM4 训练 (6000 窗口, 三预测头 type+gap+amt) ==", flush=True)
    model = MEM4().to(device)
    train(model, tr, device, epochs=20)
    torch.save(model.state_dict(), "sparkov_mem4.pt")

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, tr, device)
    pcs = per_position_ce(model, test_wins, device)
    zt = bucket_norm(pcs_tr, pcs, "t")
    zg = bucket_norm(pcs_tr, pcs, "g")
    za = bucket_norm(pcs_tr, pcs, "a")

    combos = {"仅type": zt, "仅amt": za, "type+amt": [a+b for a,b in zip(zt,za)],
              "type+amt+gap": [a+b+c for a,b,c in zip(zt,za,zg)]}
    best_name, best_auc, best_s, best_z = None, 0, None, None
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"z-norm top-{k} ({name})", y, s)
            if auc > best_auc:
                best_name, best_auc, best_s, best_z = f"top-{k} {name}", auc, s, z

    print(f"\n== 定位能力 (黑窗口内逐笔, 变体: {best_name}) ==", flush=True)
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(best_z[i].tolist())
            pos_y.extend(w["fr"].tolist())
    pos_y = np.array(pos_y)
    print(f"  黑窗口内交易 {len(pos_y)} 笔, 欺诈 {int(pos_y.sum())} 笔")
    print(f"  位置级 AUC = {roc_auc_score(pos_y, np.array(pos_z)):.4f}")

    np.savez("sparkov_scores.npz", score=best_s, label=y)
    print("\n完成 → sparkov_scores.npz", flush=True)


if __name__ == "__main__":
    main()
