# -*- coding: utf-8 -*-
"""
IBM AML v2: 加"对手方新旧"字段 —— 序列字段偷渡图信号。
洗钱模式(分散转入-集中转出)的对手方几乎全是低复现地址。
对照: v1(无对手方) = 0.656, oracle = 0.695。
"""
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from aml_experiment import (find_csv, GAP_EDGES, GAP_BOS, N_GAP, N_AMOUNT,
                            MAX_LEN, MIN_TX, bucket_norm, topk_mean, SEED)
from eth_v3 import (MEM4C, to_tensors, train as train_c,
                    per_position_ce, report)
import eth_v3

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
N_CNOV = 4


def load_accounts():
    df = pd.read_csv(find_csv(), usecols=["Timestamp", "Account", "Account.1",
                                          "Amount Paid", "Payment Format",
                                          "Is Laundering"])
    df.columns = ["ts", "src", "dst", "amt", "fmt", "y"]
    df["ts"] = pd.to_datetime(df["ts"]).astype("int64") // 10**9
    fmts = df["fmt"].value_counts().index.tolist()[:7]
    fmap = {f: i for i, f in enumerate(fmts)}
    df["fi"] = df["fmt"].map(fmap).fillna(len(fmts)).astype("int8")
    n_fmt = len(fmts) + 1

    dirty = set(df.loc[df.y == 1, "src"]) | set(df.loc[df.y == 1, "dst"])
    cnt = pd.concat([df["src"], df["dst"]]).value_counts()
    pool = set(cnt[cnt >= MIN_TX].index)
    black = pool & dirty
    white_pool = list(pool - dirty)
    rng = random.Random(SEED)
    rng.shuffle(white_pool)
    sel = black | set(white_pool[:9000])

    a = df[df["dst"].isin(sel)][["dst", "ts", "fi", "amt", "src"]]
    a = a.rename(columns={"dst": "acct", "src": "cp"}); a["dir"] = 0
    b = df[df["src"].isin(sel)][["src", "ts", "fi", "amt", "dst"]]
    b = b.rename(columns={"src": "acct", "dst": "cp"}); b["dir"] = 1
    ev = pd.concat([a, b], ignore_index=True)
    ev.sort_values(["acct", "ts"], inplace=True, kind="mergesort")

    users = []
    for acct, g in ev.groupby("acct", sort=False):
        if len(g) < MIN_TX:
            continue
        seen = defaultdict(int)
        txs = []
        for ts, fi, amt, cp, d in zip(g["ts"], g["fi"], g["amt"],
                                      g["cp"], g["dir"]):
            txs.append([float(ts), int(fi) * 2 + int(d), float(amt),
                        seen[cp]])
            seen[cp] += 1
        users.append({"addr": str(acct), "label": int(acct in black),
                      "txs": txs[-MAX_LEN:]})
    return users, n_fmt * 2


def main():
    device = "cpu"
    users, n_types = load_accounts()
    # 复用 eth_v3 的模型/编码框架, 只需改类型数
    eth_v3.N_TYPES = n_types
    normal = [u for u in users if u["label"] == 0]
    black = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(normal); rng.shuffle(black)
    black = black[:2500]
    enc = eth_v3.build_encoder(normal[:5000])
    train_u = [enc(u) for u in normal[:5000]]
    test_u = ([enc(u) for u in normal[5000:6500]] + [enc(u) for u in black])
    y = np.array([u["label"] for u in test_u])
    print(f"训练 {len(train_u)} | 测试 {int((y==0).sum())}白+{int(y.sum())}黑",
          flush=True)
    fr = [float(np.mean(np.array(u["cnov"]) == 1)) for u in test_u]
    print(f"裸特征摸底: 首次对手方占比 AUC = "
          f"{roc_auc_score(y, np.array(fr)):.4f}\n", flush=True)

    class MEM4C_AML(MEM4C):
        def __init__(self):
            nn.Module.__init__(self)
            from mem_experiment import MEM
            MEM.__init__(self, d=64, n_types=n_types, n_amount=N_AMOUNT,
                         n_gap=N_GAP, max_len=MAX_LEN)
            self.cnov_emb = nn.Embedding(N_CNOV, 8)
            self.in_proj2 = nn.Linear(32 + 8 + 16 + 8 + 8, 64)
            self.head_amt = nn.Linear(64, N_AMOUNT)
            self.head_cnov = nn.Linear(64, N_CNOV)

    print("== MEM4C 训练 (+cnov) ==", flush=True)
    model = MEM4C_AML().to(device)
    train_c(model, train_u, device, epochs=20)
    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, train_u, device)
    pcs = per_position_ce(model, test_u, device)
    z = {k: bucket_norm(pcs_tr, pcs, k) for k in ("t", "g", "a", "c")}
    combos = {"仅type": z["t"], "仅amt": z["a"], "仅cnov": z["c"],
              "type+cnov": [a+b for a, b in zip(z["t"], z["c"])],
              "SUM": [a+b+c+d for a, b, c, d in
                      zip(z["t"], z["g"], z["a"], z["c"])]}
    best = (0, None)
    for name, zz in combos.items():
        for agg, fn in [("mean", lambda a: float(a.mean())),
                        ("top5", lambda a: topk_mean(a, 5)),
                        ("top10", lambda a: topk_mean(a, 10))]:
            s = np.array([fn(a) for a in zz])
            auc = report(f"{agg} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"{agg} {name}")
    print(f"\nv1(无对手方) = 0.656 | v2(+cnov) 最优 = {best[0]:.4f} "
          f"({best[1]})", flush=True)


if __name__ == "__main__":
    main()
