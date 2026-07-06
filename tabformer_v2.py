# -*- coding: utf-8 -*-
"""
实验② v2: TabFormer + Use Chip 字段(swipe/chip/online)
------------------------------------------------------
v1 弱(窗口级 0.58)的候选原因: 编码丢了渠道字段, 而该数据集欺诈与
online 交易强相关。v2 验证: 弱是缺字段, 还是欺诈本无序列结构。
新增: ① chip 渠道进编码 ② 含 chip 的监督 LR oracle(天花板)
     ③ 用户级聚合评估(用户分 = 其窗口分位数)
"""
import math
import pickle
import random
from collections import Counter, defaultdict

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

WIN = 64
N_TYPES = 48
N_AMOUNT = 8
N_CHIP = 3
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_BOS = len(GAP_EDGES) + 1
N_GAP = GAP_BOS + 1
CACHE = "tabformer_windows.pkl"


# ---------- 数据 ----------
def build_cache():
    cols = ["User", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
            "MCC", "Is Fraud?"]
    chip_map = {"Swipe Transaction": 0, "Chip Transaction": 1,
                "Online Transaction": 2}
    chunks = []
    for ch in pd.read_csv("card_transaction.v1.csv", usecols=cols,
                          chunksize=2_000_000):
        hm = ch["Time"].str.split(":", expand=True).astype("int16")
        ts = (pd.to_datetime(dict(year=ch["Year"], month=ch["Month"],
                                  day=ch["Day"])).astype("int64") // 10**9
              + hm[0] * 3600 + hm[1] * 60)
        chunks.append(pd.DataFrame({
            "user": ch["User"].astype("int32"),
            "ts": ts.astype("int64"),
            "amount": ch["Amount"].str.replace("$", "", regex=False)
                                   .astype("float32"),
            "chip": ch["Use Chip"].map(chip_map).fillna(0).astype("int8"),
            "mcc": ch["MCC"].astype("int32"),
            "fraud": (ch["Is Fraud?"] == "Yes").astype("int8")}))
    df = pd.concat(chunks, ignore_index=True)
    df.sort_values(["user", "ts"], inplace=True, kind="mergesort")

    fraud_users = set(df.loc[df["fraud"] == 1, "user"].unique())
    clean_users = [u for u in df["user"].unique() if u not in fraud_users]
    rng = random.Random(SEED)
    rng.shuffle(clean_users)
    n_tr = int(len(clean_users) * 0.8)
    tr_users, te_users = set(clean_users[:n_tr]), set(clean_users[n_tr:])

    wins = {"train": [], "test_w": [], "black": []}
    for user, g in df.groupby("user", sort=False):
        arr = {k: g[k].to_numpy() for k in ("ts", "amount", "chip", "mcc", "fraud")}
        for s in range(0, len(g) - WIN + 1, WIN):
            w = {k: v[s:s+WIN] for k, v in arr.items()}
            w["user"] = user
            if user in tr_users:
                wins["train"].append(w)
            elif user in te_users:
                wins["test_w"].append(w)
            elif w["fraud"].any():
                wins["black"].append(w)
    for k in wins:
        rng.shuffle(wins[k])
    with open(CACHE, "wb") as f:
        pickle.dump(wins, f)
    return wins


def load_windows():
    try:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return build_cache()


def encode_all(wins, n_train, n_testw, n_black):
    tr = wins["train"][:n_train]
    mcc_cnt = Counter()
    amts = []
    for w in tr:
        mcc_cnt.update(w["mcc"].tolist())
        amts.extend(w["amount"].tolist())
    vocab = {m: i for i, (m, _) in enumerate(mcc_cnt.most_common(N_TYPES - 1))}
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_AMOUNT + 1)[1:-1])

    def enc(w):
        gaps = np.maximum(np.diff(w["ts"]), 0)
        return {"types": [vocab.get(m, N_TYPES - 1) for m in w["mcc"]],
                "amounts": np.searchsorted(q, w["amount"]).tolist(),
                "gaps": [GAP_BOS] + np.searchsorted(GAP_EDGES, gaps,
                                                    side="left").tolist(),
                "hours": ((w["ts"] % 86400) / 3600.0).tolist(),
                "chips": w["chip"].tolist(),
                "label": int(w["fraud"].any()), "fr": w["fraud"],
                "user": w["user"]}

    return ([enc(w) for w in tr],
            [enc(w) for w in wins["test_w"][:n_testw]],
            [enc(w) for w in wins["black"][:n_black]])


# ---------- 模型(5 字段版) ----------
class MEM5(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.amount_emb = nn.Embedding(N_AMOUNT, 8)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.chip_emb = nn.Embedding(N_CHIP, 8)
        self.hour_proj = nn.Linear(2, 8)
        self.in_proj = nn.Linear(32 + 8 + 16 + 8 + 8, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.pos_emb = nn.Embedding(WIN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_type = nn.Linear(d, N_TYPES)
        self.head_gap = nn.Linear(d, N_GAP)
        self.head_chip = nn.Linear(d, N_CHIP)

    def forward(self, tp, am, gp, hr, cp, mask):
        x = self.in_proj(torch.cat([self.type_emb(tp), self.amount_emb(am),
                                    self.gap_emb(gp), self.hour_proj(hr),
                                    self.chip_emb(cp)], -1))
        x = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        h = self.encoder(x)
        return self.head_type(h), self.head_gap(h), self.head_chip(h)


def to_tensors(batch, device):
    tp = torch.tensor([w["types"] for w in batch], dtype=torch.long)
    am = torch.tensor([w["amounts"] for w in batch], dtype=torch.long)
    gp = torch.tensor([w["gaps"] for w in batch], dtype=torch.long)
    cp = torch.tensor([w["chips"] for w in batch], dtype=torch.long)
    h = torch.tensor([w["hours"] for w in batch])
    hr = torch.stack([torch.sin(2 * math.pi * h / 24),
                      torch.cos(2 * math.pi * h / 24)], -1)
    return (tp.to(device), am.to(device), gp.to(device), hr.to(device),
            cp.to(device))


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
            tp, am, gp, hr, cp = to_tensors(batch, device)
            mask = torch.rand(tp.shape) < ratio
            mask[:, 0] |= ~mask.any(1)
            mask = mask.to(device)
            lt, lg, lc = model(tp, am, gp, hr, cp, mask)
            loss = (ce(lt[mask], tp[mask]) + ce(lg[mask], gp[mask]) +
                    ce(lc[mask], cp[mask]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, wins, device, stride=8, bs=128):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{"t": np.zeros(WIN), "g": np.zeros(WIN), "c": np.zeros(WIN),
            "gb": np.array(w["gaps"])} for w in wins]
    for r in range(stride):
        for s in range(0, len(wins), bs):
            batch = wins[s:s + bs]
            tp, am, gp, hr, cp = to_tensors(batch, device)
            pos = torch.arange(WIN)
            mask = ((pos % stride) == r)[None].expand(len(batch), WIN)
            lt, lg, lc = model(tp, am, gp, hr, cp, mask.to(device))
            idx = mask[0].nonzero().flatten()
            for i in range(len(batch)):
                out[s + i]["t"][idx] = ce(lt[i, idx], tp[i, idx],
                                          reduction="none").cpu().numpy()
                out[s + i]["g"][idx] = ce(lg[i, idx], gp[i, idx],
                                          reduction="none").cpu().numpy()
                out[s + i]["c"][idx] = ce(lc[i, idx], cp[i, idx],
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
    wins = load_windows()
    print(f"窗口池: 训练白 {len(wins['train'])}, 测试白 {len(wins['test_w'])}, "
          f"黑 {len(wins['black'])}", flush=True)
    tr, tw, bl = encode_all(wins, 6000, 1500, 2000)
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    print(f"编码完成: 训练 {len(tr)}, 测试 {len(tw)}白+{len(bl)}黑\n", flush=True)

    # ---- 监督 oracle 天花板(含 chip) ----
    print("== 监督 oracle (5折CV, 含chip/MCC/金额全特征) ==", flush=True)
    allw = tr + test_wins
    ya = np.array([w["label"] for w in allw])
    feats = []
    for w in allw:
        mcch = np.bincount(w["types"], minlength=N_TYPES) / WIN
        chipf = np.bincount(w["chips"], minlength=N_CHIP) / WIN
        feats.append(np.concatenate([
            mcch, chipf,
            [np.mean(w["amounts"]), np.max(w["amounts"]),
             np.mean(np.array(w["gaps"]) <= 2), np.mean(w["hours"])]]))
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(lr_pipe, np.array(feats), ya, cv=5,
                          method="predict_proba")[:, 1]
    report("Oracle: MCC+chip+金额统计+LR", ya, p)
    chip_only = np.array([np.mean(np.array(w["chips"]) == 2) for w in allw])
    report("Oracle: 仅online交易占比", ya, chip_only)

    # ---- MEM5 ----
    print("\n== MEM5 训练 (6000 白窗口, 三预测目标 type+gap+chip) ==", flush=True)
    model = MEM5().to(device)
    train(model, tr, device, epochs=20)
    torch.save(model.state_dict(), "tabformer_mem5.pt")

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, tr, device)
    pcs = per_position_ce(model, test_wins, device)
    zt = bucket_norm(pcs_tr, pcs, "t")
    zg = bucket_norm(pcs_tr, pcs, "g")
    zc = bucket_norm(pcs_tr, pcs, "c")

    combos = {"仅type": zt, "仅chip": zc,
              "type+chip": [a + b for a, b in zip(zt, zc)],
              "type+gap+chip": [a + b + c for a, b, c in zip(zt, zg, zc)]}
    best_name, best_auc, best_s = None, 0, None
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"z-norm top-{k} ({name})", y, s)
            if auc > best_auc:
                best_name, best_auc, best_s = f"top-{k} {name}", auc, s

    # ---- 用户级聚合 ----
    print(f"\n== 用户级聚合 (窗口分取 max, 最优变体: {best_name}) ==", flush=True)
    user_scores = defaultdict(list)
    user_label = {}
    for i, w in enumerate(test_wins):
        user_scores[w["user"]].append(best_s[i])
        user_label[w["user"]] = w["label"]
    uy = np.array([user_label[u] for u in user_scores])
    us = np.array([max(v) for v in user_scores.values()])
    print(f"  用户数: 白 {int((uy==0).sum())}, 黑 {int((uy==1).sum())}")
    report("用户级 AUC", uy, us)

    # ---- 定位能力 ----
    print("\n== 定位能力 (黑窗口内逐交易) ==", flush=True)
    z_best = combos["type+chip"]
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(z_best[i].tolist())
            pos_y.extend(w["fr"].tolist())
    print(f"  位置级 AUC (type+chip) = "
          f"{roc_auc_score(np.array(pos_y), np.array(pos_z)):.4f}")
    pos_z2 = []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z2.extend(zt[i].tolist())
    print(f"  位置级 AUC (仅type)    = "
          f"{roc_auc_score(np.array(pos_y), np.array(pos_z2)):.4f}")


if __name__ == "__main__":
    main()
