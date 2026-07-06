# -*- coding: utf-8 -*-
"""
实验②: 真实数据验证 —— IBM TabFormer 信用卡交易数据集
------------------------------------------------------
~2000 用户 / ~2400 万笔真实感交易 / 逐笔欺诈标注(约 0.12%)。
按 User 聚合成时间有序的交易序列, 切成长度 64 的不重叠窗口:
  白窗口 = 无任何欺诈交易的用户的窗口(按用户切 train/test, 防泄漏)
  黑窗口 = 欺诈用户中"包含≥1笔欺诈交易"的窗口
MEM 只用白窗口(训练用户)训练, 打分方式同 temporal 实验(z-norm top-k)。

编码: MCC top-47+OTHER 为事件类型, 金额 8 分位桶, 间隔桶同前, 小时 sin/cos。
对照: MCC 直方图+LR、金额统计+LR(oracle, 量化"词频/边缘特征"在真实数据的含金量)。
额外验证: 黑窗口内逐位置 z 分 vs 是否欺诈交易 —— 定位能力。
"""
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mem_experiment import MEM, train, SEED
from mem_score_v2 import bucket_stats, topk_mean

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WIN = 64
N_TYPES = 48          # MCC top-47 + OTHER
N_AMOUNT = 8
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_BOS = len(GAP_EDGES) + 1
N_GAP = GAP_BOS + 1
N_TRAIN_WIN = 4000
N_TESTW_WIN = 1500
N_BLACK_WIN = 2000


def load_transactions(path="card_transaction.v1.csv"):
    cols = ["User", "Year", "Month", "Day", "Time", "Amount", "MCC", "Is Fraud?"]
    chunks = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=2_000_000):
        ch["Amount"] = ch["Amount"].str.replace("$", "", regex=False).astype("float32")
        hm = ch["Time"].str.split(":", expand=True).astype("int16")
        ts = (pd.to_datetime(dict(year=ch["Year"], month=ch["Month"], day=ch["Day"]))
              .astype("int64") // 10**9 + hm[0] * 3600 + hm[1] * 60)
        chunks.append(pd.DataFrame({
            "user": ch["User"].astype("int32"),
            "ts": ts.astype("int64"),
            "amount": ch["Amount"],
            "mcc": ch["MCC"].astype("int32"),
            "fraud": (ch["Is Fraud?"] == "Yes").astype("int8")}))
    df = pd.concat(chunks, ignore_index=True)
    df.sort_values(["user", "ts"], inplace=True, kind="mergesort")
    return df


def gap_bucket_arr(gaps):
    b = np.searchsorted(GAP_EDGES, gaps, side="left")
    return b


def build_windows(df):
    fraud_users = set(df.loc[df["fraud"] == 1, "user"].unique())
    all_users = df["user"].unique()
    clean_users = [u for u in all_users if u not in fraud_users]
    print(f"用户: {len(all_users)} 总, {len(clean_users)} 干净, {len(fraud_users)} 含欺诈")
    print(f"交易: {len(df)} 总, 欺诈 {int(df['fraud'].sum())} "
          f"({df['fraud'].mean():.3%})")

    rng = random.Random(SEED)
    rng.shuffle(clean_users)
    n_tr = int(len(clean_users) * 0.8)
    tr_users, te_users = set(clean_users[:n_tr]), set(clean_users[n_tr:])

    wins = {"train": [], "test_w": [], "black": []}
    for user, g in df.groupby("user", sort=False):
        ts = g["ts"].to_numpy()
        amt = g["amount"].to_numpy()
        mcc = g["mcc"].to_numpy()
        fr = g["fraud"].to_numpy()
        for s in range(0, len(g) - WIN + 1, WIN):
            w = {"ts": ts[s:s+WIN], "amt": amt[s:s+WIN],
                 "mcc": mcc[s:s+WIN], "fr": fr[s:s+WIN], "user": user}
            if user in tr_users:
                wins["train"].append(w)
            elif user in te_users:
                wins["test_w"].append(w)
            elif w["fr"].any():
                wins["black"].append(w)
    print(f"窗口: 训练白 {len(wins['train'])}, 测试白 {len(wins['test_w'])}, "
          f"黑 {len(wins['black'])}")
    rng.shuffle(wins["train"]); rng.shuffle(wins["test_w"]); rng.shuffle(wins["black"])
    wins["train"] = wins["train"][:N_TRAIN_WIN]
    wins["test_w"] = wins["test_w"][:N_TESTW_WIN]
    wins["black"] = wins["black"][:N_BLACK_WIN]
    return wins


def encode_windows(wins):
    """确定 MCC 词表与金额分位桶(仅用训练窗口), 编码为 MEM 输入格式"""
    mcc_cnt = Counter()
    amts = []
    for w in wins["train"]:
        mcc_cnt.update(w["mcc"].tolist())
        amts.extend(w["amt"].tolist())
    vocab = {m: i for i, (m, _) in enumerate(mcc_cnt.most_common(N_TYPES - 1))}
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_AMOUNT + 1)[1:-1])

    def enc(w):
        types = [vocab.get(m, N_TYPES - 1) for m in w["mcc"]]
        amounts = np.searchsorted(q, w["amt"]).tolist()
        gaps = np.diff(w["ts"])
        gb = [GAP_BOS] + gap_bucket_arr(np.maximum(gaps, 0)).tolist()
        hours = ((w["ts"] % 86400) / 3600.0).tolist()
        return {"types": types, "amounts": amounts, "gaps": gb, "hours": hours,
                "label": int(w["fr"].any()), "fr": w["fr"], "user": w["user"]}

    return {k: [enc(w) for w in v] for k, v in wins.items()}, vocab, q


@torch.no_grad()
def per_position_ce(model, users, device, stride=8, bs=64):
    import torch.nn as nn
    from mem_experiment import pad_batch
    model.eval()
    out = [{"t": np.zeros(len(u["types"])), "g": np.zeros(len(u["types"])),
            "gb": np.array(u["gaps"]), "tp": np.array(u["types"])}
           for u in users]
    ce = nn.functional.cross_entropy
    for r in range(stride):
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            tp, am, gp, hr, pad = pad_batch(batch, device)
            L = tp.size(1)
            pos = torch.arange(L)
            mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
            if not mask.any():
                continue
            lt, lg = model(tp, am, gp, hr, pad, mask.to(device))
            for i in range(len(batch)):
                idx = mask[i].nonzero().flatten()
                if len(idx) == 0:
                    continue
                out[s + i]["t"][idx.numpy()] = ce(
                    lt[i, idx], tp[i, idx], reduction="none").cpu().numpy()
                out[s + i]["g"][idx.numpy()] = ce(
                    lg[i, idx], gp[i, idx], reduction="none").cpu().numpy()
    return out


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return (roc_auc_score(y, s), float(np.max(tpr - fpr)),
            float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]))


def report(name, y, s):
    auc, ks, r1 = metrics(y, s)
    print(f"  {name:<34} AUC={auc:.4f}  KS={ks:.4f}  R@FPR1%={r1:.1%}")


def main():
    device = "cpu"
    print("== 加载真实交易数据 ==")
    df = load_transactions()
    wins = build_windows(df)
    del df
    enc, vocab, q = encode_windows(wins)
    train_wins = enc["train"]
    test_wins = enc["test_w"] + enc["black"]
    y = np.array([w["label"] for w in test_wins])
    print(f"测试: {len(enc['test_w'])} 白窗口 + {len(enc['black'])} 黑窗口\n")

    # ---- Oracle 基线: 边缘特征含金量 ----
    print("== Oracle 基线 (5折CV, 训练白+测试全体) ==")
    allw = train_wins + test_wins
    ya = np.array([w["label"] for w in allw])
    mcch = np.zeros((len(allw), N_TYPES))
    for i, w in enumerate(allw):
        for t in w["types"]:
            mcch[i, t] += 1 / WIN
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(lr_pipe, mcch, ya, cv=5, method="predict_proba")[:, 1]
    report("B1 MCC直方图+LR", ya, p)
    astat = np.array([[np.mean(w["amounts"]), np.max(w["amounts"]),
                       np.mean(np.array(w["gaps"]) <= 2),
                       np.mean(w["hours"])] for w in allw])
    p = cross_val_predict(lr_pipe, astat, ya, cv=5, method="predict_proba")[:, 1]
    report("B2 金额/间隔/时段统计+LR", ya, p)

    # ---- MEM ----
    print("\n== MEM 训练 (仅白窗口) ==")
    model = MEM(n_types=N_TYPES, n_amount=N_AMOUNT).to(device)
    train(model, train_wins, device, epochs=15, bs=64)
    torch.save(model.state_dict(), "tabformer_mem.pt")

    print("\n== 打分 ==")
    pcs_tr = per_position_ce(model, train_wins, device)
    stats = bucket_stats(pcs_tr)
    mu_t, sd_t, mu_g, sd_g = stats
    pcs = per_position_ce(model, test_wins, device)

    def zt_zg(p):
        b = np.clip(p["gb"], 0, N_GAP - 1)
        return ((p["t"] - mu_t[b]) / sd_t[b], (p["g"] - mu_g[b]) / sd_g[b])

    zs = [zt_zg(p) for p in pcs]
    report("raw 平均", y, np.array([(p["t"] + p["g"]).mean() for p in pcs]))
    report("z-norm 平均(type+gap)", y,
           np.array([(zt + zg).mean() for zt, zg in zs]))
    for k in (3, 5, 10):
        report(f"z-norm top-{k}(仅type)", y,
               np.array([topk_mean(zt, k) for zt, _ in zs]))
        report(f"z-norm top-{k}(type+gap)", y,
               np.array([topk_mean(zt + zg, k) for zt, zg in zs]))

    # ---- 逐位置定位能力: 黑窗口内, 欺诈交易的 z 分更高吗? ----
    print("\n== 定位能力 (黑窗口内逐交易: z分 vs 是否欺诈) ==")
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            zt, zg = zs[i]
            pos_z.extend((zt + zg).tolist())
            pos_y.extend(w["fr"].tolist())
    pos_y = np.array(pos_y)
    print(f"  黑窗口内交易数 {len(pos_y)}, 其中欺诈 {int(pos_y.sum())}")
    print(f"  位置级 AUC = {roc_auc_score(pos_y, np.array(pos_z)):.4f}")

    np.savez("tabformer_scores.npz",
             score=np.array([topk_mean(zt + zg, 5) for zt, zg in zs]), label=y)


if __name__ == "__main__":
    main()
