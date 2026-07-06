# -*- coding: utf-8 -*-
"""
开源数据验证 #4: PaySim (Kaggle ealaxi/paysim1) — 收款账户(骡子账户)视角
------------------------------------------------------------------------
发起方账户已实测不可行(中位出现1次)。改为收款账户任务:
  黑 = 收到过 isFraud 交易的账户(骗子控制的收款账户, ~2416 个交易≥10笔)
  白 = 其他收款≥10笔的账户
事件 = 该账户参与的每笔交易: [交易类型×方向(10类), 金额档, 间隔桶(小时), 时刻]
配方: 以太坊配方(整事件遮罩 + mean/top-k 双通道) —— 资金流形状。
已知伪影: PaySim 欺诈 TRANSFER 金额常=全余额, oracle 基线会量化金额泄漏。
"""
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mem_experiment import MEM, pad_batch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CSV = ("/Users/macbookpro/.cache/kagglehub/datasets/ealaxi/paysim1/"
       "versions/2/PS_20174392719_1491204439457_log.csv")
TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
N_TYPES = 10          # 5 类型 × 2 方向
N_AMOUNT = 8
GAP_EDGES_H = [1, 3, 6, 12, 24, 72, 168]   # 小时粒度
GAP_BOS = len(GAP_EDGES_H) + 1
N_GAP = GAP_BOS + 1
MAX_LEN = 120
MIN_TX = 10


def load_accounts():
    df = pd.read_csv(CSV, usecols=["step", "nameOrig", "nameDest", "type",
                                   "amount", "isFraud"])
    tmap = {t: i for i, t in enumerate(TX_TYPES)}
    df["ty"] = df["type"].map(tmap).astype("int8")

    vcd = df["nameDest"].value_counts()
    dest_pool = set(vcd[vcd >= MIN_TX].index)
    fraud_dest = set(df.loc[df["isFraud"] == 1, "nameDest"].unique())
    black = dest_pool & fraud_dest
    white_pool = list(dest_pool - fraud_dest)
    rng = random.Random(SEED)
    rng.shuffle(white_pool)
    white = set(white_pool[:8000])
    sel = black | white
    print(f"账户: 黑 {len(black)}, 白采样 {len(white)}", flush=True)

    # 该账户参与的所有交易(作为收款方 dir=0 / 作为发起方 dir=1)
    a = df[df["nameDest"].isin(sel)][["nameDest", "step", "ty", "amount"]]
    a = a.rename(columns={"nameDest": "acct"}); a["dir"] = 0
    b = df[df["nameOrig"].isin(sel)][["nameOrig", "step", "ty", "amount"]]
    b = b.rename(columns={"nameOrig": "acct"}); b["dir"] = 1
    ev = pd.concat([a, b], ignore_index=True)
    ev.sort_values(["acct", "step"], inplace=True, kind="mergesort")

    users = []
    for acct, g in ev.groupby("acct", sort=False):
        if len(g) < MIN_TX:
            continue
        users.append({"uid": acct, "label": int(acct in black),
                      "step": g["step"].to_numpy()[-MAX_LEN:],
                      "ty": g["ty"].to_numpy()[-MAX_LEN:],
                      "dir": g["dir"].to_numpy()[-MAX_LEN:],
                      "amount": g["amount"].to_numpy()[-MAX_LEN:]})
    return users


def build_encoder(train_users):
    amts = np.concatenate([u["amount"] for u in train_users])
    amts = amts[amts > 0]
    q = np.quantile(amts, np.linspace(0, 1, N_AMOUNT)[1:-1])

    def enc(u):
        types = (u["ty"] * 2 + u["dir"]).tolist()
        amounts = [0 if a <= 0 else 1 + int(np.searchsorted(q, a))
                   for a in u["amount"]]
        gaps, prev = [], None
        for s in u["step"]:
            gaps.append(GAP_BOS if prev is None else
                        int(np.searchsorted(GAP_EDGES_H, max(s - prev, 0),
                                            side="left")))
            prev = s
        hours = (u["step"] % 24).astype(float).tolist()
        return {"uid": u["uid"], "label": u["label"], "types": types,
                "amounts": amounts, "gaps": gaps, "hours": hours}
    return enc


class MEM3(MEM):
    def __init__(self):
        super().__init__(d=64, n_types=N_TYPES, n_amount=N_AMOUNT,
                         n_gap=N_GAP, max_len=MAX_LEN)
        self.head_amt = nn.Linear(64, N_AMOUNT)

    def forward(self, tp, am, gp, hr, pad, mask):
        h = self.encode(tp, am, gp, hr, pad, mask)
        return self.head_type(h), self.head_gap(h), self.head_amt(h)


def random_mask(pad, ratio, gen):
    scores = torch.rand(pad.shape, generator=gen)
    scores[pad] = -1.0
    mask = scores > (1 - ratio)
    for i in range(pad.size(0)):
        if not mask[i].any():
            valid = (~pad[i]).nonzero().flatten()
            mask[i, valid[torch.randint(len(valid), (1,), generator=gen)]] = True
    return mask


def train(model, users, device, epochs=20, bs=64, lr=1e-3, ratio=0.15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.functional.cross_entropy
    gen = torch.Generator().manual_seed(SEED)
    model.train()
    for ep in range(epochs):
        order = list(range(len(users)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [users[j] for j in order[s:s + bs]]
            tp, am, gp, hr, pad = pad_batch(batch, device)
            mask = random_mask(pad, ratio, gen).to(device)
            lt, lg, la = model(tp, am, gp, hr, pad, mask)
            loss = (ce(lt[mask], tp[mask]) + ce(lg[mask], gp[mask]) +
                    ce(la[mask], am[mask]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, users, device, stride=7, bs=64):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{"t": np.zeros(len(u["types"])), "g": np.zeros(len(u["types"])),
            "a": np.zeros(len(u["types"])), "gb": np.array(u["gaps"])}
           for u in users]
    for r in range(stride):
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            tp, am, gp, hr, pad = pad_batch(batch, device)
            pos = torch.arange(tp.size(1))
            mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
            if not mask.any():
                continue
            lt, lg, la = model(tp, am, gp, hr, pad, mask.to(device))
            for i in range(len(batch)):
                idx = mask[i].nonzero().flatten()
                if len(idx) == 0:
                    continue
                out[s+i]["t"][idx.numpy()] = ce(lt[i, idx], tp[i, idx],
                                                reduction="none").cpu().numpy()
                out[s+i]["g"][idx.numpy()] = ce(lg[i, idx], gp[i, idx],
                                                reduction="none").cpu().numpy()
                out[s+i]["a"][idx.numpy()] = ce(la[i, idx], am[i, idx],
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


def report(name, y, s):
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    print(f"  {name:<32} AUC={auc:.4f}  KS={np.max(tpr-fpr):.4f}  "
          f"R@FPR1%={r1:.1%}", flush=True)
    return auc


def main():
    device = "cpu"
    print("== 加载 PaySim, 构建收款账户序列 ==", flush=True)
    users = load_accounts()
    normal = [u for u in users if u["label"] == 0]
    black = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(normal)
    print(f"序列化后: 正常 {len(normal)}, 欺诈收款 {len(black)}", flush=True)
    enc = build_encoder(normal[:5000])
    train_u = [enc(u) for u in normal[:5000]]
    test_u = [enc(u) for u in normal[5000:6500]] + [enc(u) for u in black]
    y = np.array([u["label"] for u in test_u])
    print(f"训练 {len(train_u)} | 测试 {int((y==0).sum())}白 + {int(y.sum())}黑\n",
          flush=True)

    print("== 监督 oracle (5折CV, 账户级统计) ==", flush=True)
    allu = train_u + test_u
    ya = np.array([u["label"] for u in allu])
    feats = []
    for u in allu:
        th = np.bincount(u["types"], minlength=N_TYPES) / len(u["types"])
        g = np.array(u["gaps"][1:]) if len(u["gaps"]) > 1 else np.array([8])
        feats.append(np.concatenate([th, [len(u["types"]),
                     np.mean(u["amounts"]), np.max(u["amounts"]),
                     np.mean(g <= 1)]]))
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(lr_pipe, np.array(feats), ya, cv=5,
                          method="predict_proba")[:, 1]
    report("Oracle: 类型比/金额/密集度+LR", ya, p)

    print("\n== MEM3 训练 (5000 正常收款账户) ==", flush=True)
    model = MEM3().to(device)
    train(model, train_u, device)
    torch.save(model.state_dict(), "paysim_mem3.pt")

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, train_u, device)
    pcs = per_position_ce(model, test_u, device)
    zt = bucket_norm(pcs_tr, pcs, "t")
    zg = bucket_norm(pcs_tr, pcs, "g")
    za = bucket_norm(pcs_tr, pcs, "a")
    combos = {"仅type": zt, "仅amt": za, "仅gap": zg,
              "type+amt": [a+b for a, b in zip(zt, za)],
              "type+amt+gap": [a+b+c for a, b, c in zip(zt, za, zg)]}
    best = (0, None)
    for name, z in combos.items():
        for agg, fn in [("mean", lambda a: a.mean()),
                        ("top5", lambda a: topk_mean(a, 5))]:
            s = np.array([fn(a) for a in z])
            auc = report(f"z-norm {agg} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"{agg} {name}")
    print(f"\n最优: {best[1]}  AUC={best[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
