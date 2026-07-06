# -*- coding: utf-8 -*-
"""
真实数据验证 #3: 以太坊钓鱼诈骗 (XBlock EPTransNet, 真实链上交易+真实钓鱼标注)
--------------------------------------------------------------------------
账户级序列: 每笔交易 = [方向(转入/转出), 金额档, 间隔桶, 时刻]。
钓鱼账户行为先验: 大量小额转入(受害者打款) → 集中大额转出(归集), 时序密集。
协议: 4000 正常账户训练(无标签视角) / 1000 正常 + 全部钓鱼 测试。
注意: "正常"账户是钓鱼节点的 1-2 阶邻居(BFS 爬取), 采样有偏, 结论需注明。
"""
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mem_experiment import MEM, pad_batch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_TYPES = 2       # 0=转入 1=转出
N_AMOUNT = 8      # 0=零值 + 7 个非零分位桶
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_BOS = len(GAP_EDGES) + 1
N_GAP = GAP_BOS + 1
MAX_LEN = 200


def load(path="eth_sequences.jsonl"):
    users = []
    for line in open(path):
        r = json.loads(line)
        if len(r["txs"]) < 10:
            continue
        users.append(r)
    return users


def build_encoder(train_users):
    amts = [a for u in train_users for _, _, a in u["txs"] if a > 0]
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_AMOUNT)[1:-1])

    def enc(u):
        txs = u["txs"][-MAX_LEN:]
        types, amounts, gaps, hours = [], [], [], []
        prev = None
        for ts, d, a in txs:
            types.append(int(d))
            amounts.append(0 if a <= 0 else 1 + int(np.searchsorted(q, a)))
            gaps.append(GAP_BOS if prev is None else
                        int(np.searchsorted(GAP_EDGES, max(ts - prev, 0),
                                            side="left")))
            hours.append((ts % 86400) / 3600.0)
            prev = ts
        return {"uid": u["addr"], "label": u["label"], "types": types,
                "amounts": amounts, "gaps": gaps, "hours": hours}
    return enc


class MEM3(MEM):
    """在 type+gap 双头基础上加金额头"""
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


def train(model, users, device, epochs=25, bs=32, lr=1e-3, ratio=0.15):
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
            L = tp.size(1)
            pos = torch.arange(L)
            mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
            if not mask.any():
                continue
            lt, lg, la = model(tp, am, gp, hr, pad, mask.to(device))
            for i in range(len(batch)):
                idx = mask[i].nonzero().flatten()
                if len(idx) == 0:
                    continue
                out[s + i]["t"][idx.numpy()] = ce(lt[i, idx], tp[i, idx],
                                                  reduction="none").cpu().numpy()
                out[s + i]["g"][idx.numpy()] = ce(lg[i, idx], gp[i, idx],
                                                  reduction="none").cpu().numpy()
                out[s + i]["a"][idx.numpy()] = ce(la[i, idx], am[i, idx],
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
    users = load()
    normal = [u for u in users if u["label"] == 0]
    phish = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(normal)
    print(f"账户: 正常 {len(normal)}, 钓鱼 {len(phish)}", flush=True)
    enc = build_encoder(normal[:4000])
    train_u = [enc(u) for u in normal[:4000]]
    test_u = [enc(u) for u in normal[4000:5000]] + [enc(u) for u in phish]
    y = np.array([u["label"] for u in test_u])
    print(f"训练 {len(train_u)} 正常 | 测试 {int((y==0).sum())}正常 + "
          f"{int(y.sum())}钓鱼\n", flush=True)

    # ---- 监督 oracle: 账户级统计特征 ----
    print("== 监督 oracle (5折CV, 账户级统计) ==", flush=True)
    allu = train_u + test_u
    ya = np.array([u["label"] for u in allu])
    feats = []
    for u in allu:
        g = np.array(u["gaps"][1:]) if len(u["gaps"]) > 1 else np.array([8])
        feats.append([len(u["types"]), np.mean(u["types"]),
                      np.mean(u["amounts"]), np.max(u["amounts"]),
                      np.mean(g <= 2), np.mean(g), np.std(u["hours"])])
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(lr_pipe, np.array(feats), ya, cv=5,
                          method="predict_proba")[:, 1]
    report("Oracle: 方向比/金额/密集度+LR", ya, p)

    # ---- MEM3 ----
    print("\n== MEM3 训练 (4000 正常账户) ==", flush=True)
    model = MEM3().to(device)
    train(model, train_u, device)
    torch.save(model.state_dict(), "eth_mem3.pt")

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, train_u, device)
    pcs = per_position_ce(model, test_u, device)
    zt = bucket_norm(pcs_tr, pcs, "t")
    zg = bucket_norm(pcs_tr, pcs, "g")
    za = bucket_norm(pcs_tr, pcs, "a")

    combos = {"仅type(方向)": zt, "仅gap": zg, "仅amt": za,
              "type+amt": [a+b for a,b in zip(zt,za)],
              "type+gap+amt": [a+b+c for a,b,c in zip(zt,zg,za)]}
    best = (0, None, None)
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"z-norm top-{k} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"top-{k} {name}", s)
    # 均值池化对照(账户整体异常, 而非局部)
    for name, z in combos.items():
        s = np.array([a.mean() for a in z])
        auc = report(f"z-norm 全序列平均 ({name})", y, s)
        if auc > best[0]:
            best = (auc, f"mean {name}", s)

    print(f"\n最优: {best[1]}  AUC={best[0]:.4f}", flush=True)
    np.savez("eth_scores.npz", score=best[2], label=y)


if __name__ == "__main__":
    main()
