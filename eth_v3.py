# -*- coding: utf-8 -*-
"""
以太坊 v3: 加"对手方新旧"字段(cnov) —— 用序列字段偷渡图信号。
钓鱼收款账户的特征: 打款方几乎全是首次出现的陌生地址。
cnov 桶: 1=首次对手方, 2=第2-3次, 3=第4次及以上 (0 保留)。
对照: v1(无对手方信息) = 0.904, oracle = 0.934。
输入 eth_sequences2.jsonl (含 prior 计数)。
"""
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from mem_experiment import MEM, pad_batch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

N_TYPES, N_AMOUNT, N_CNOV = 2, 8, 4
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_BOS = len(GAP_EDGES) + 1
N_GAP = GAP_BOS + 1
MAX_LEN = 200


def load(path="eth_sequences2.jsonl"):
    users = []
    for line in open(path):
        r = json.loads(line)
        if len(r["txs"]) >= 10:
            users.append(r)
    return users


def build_encoder(train_users):
    amts = [a for u in train_users for _, _, a, _ in u["txs"] if a > 0]
    q = np.quantile(np.array(amts), np.linspace(0, 1, N_AMOUNT)[1:-1])

    def enc(u):
        txs = u["txs"][-MAX_LEN:]
        o = {"uid": u["addr"], "label": u["label"], "types": [],
             "amounts": [], "gaps": [], "hours": [], "cnov": []}
        prev = None
        for ts, d, a, prior in txs:
            o["types"].append(int(d))
            o["amounts"].append(0 if a <= 0 else 1 + int(np.searchsorted(q, a)))
            o["gaps"].append(GAP_BOS if prev is None else
                             int(np.searchsorted(GAP_EDGES, max(ts - prev, 0),
                                                 side="left")))
            o["hours"].append((ts % 86400) / 3600.0)
            o["cnov"].append(1 if prior == 0 else (2 if prior <= 2 else 3))
            prev = ts
        return o
    return enc


class MEM4C(MEM):
    """type/amt/gap/hour + cnov 字段, 四预测头(type/gap/amt/cnov)"""
    def __init__(self):
        super().__init__(d=64, n_types=N_TYPES, n_amount=N_AMOUNT,
                         n_gap=N_GAP, max_len=MAX_LEN)
        self.cnov_emb = nn.Embedding(N_CNOV, 8)
        self.in_proj2 = nn.Linear(32 + 8 + 16 + 8 + 8, 64)
        self.head_amt = nn.Linear(64, N_AMOUNT)
        self.head_cnov = nn.Linear(64, N_CNOV)

    def forward(self, tp, am, gp, hr, cn, pad, mask):
        x = self.in_proj2(torch.cat(
            [self.type_emb(tp), self.amount_emb(am), self.gap_emb(gp),
             self.hour_proj(torch.stack([torch.sin(2*math.pi*hr/24),
                                         torch.cos(2*math.pi*hr/24)], -1)),
             self.cnov_emb(cn)], -1))
        x = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        h = self.encoder(x, src_key_padding_mask=pad)
        return (self.head_type(h), self.head_gap(h),
                self.head_amt(h), self.head_cnov(h))


def to_tensors(batch, device):
    n = len(batch)
    L = max(len(u["types"]) for u in batch)
    T = {}
    for f in ("types", "amounts", "gaps", "cnov"):
        M = torch.zeros(n, L, dtype=torch.long)
        for i, u in enumerate(batch):
            M[i, :len(u[f])] = torch.tensor(u[f])
        T[f] = M.to(device)
    H = torch.zeros(n, L)
    pad = torch.ones(n, L, dtype=torch.bool)
    for i, u in enumerate(batch):
        H[i, :len(u["hours"])] = torch.tensor(u["hours"])
        pad[i, :len(u["hours"])] = False
    return T, H.to(device), pad.to(device)


def random_mask(pad, ratio, gen):
    sc = torch.rand(pad.shape, generator=gen); sc[pad] = -1
    mask = sc > (1 - ratio)
    for i in range(pad.size(0)):
        if not mask[i].any():
            v = (~pad[i]).nonzero().flatten()
            mask[i, v[torch.randint(len(v), (1,), generator=gen)]] = True
    return mask


def train(model, users, device, epochs=25, bs=32, lr=1e-3, ratio=0.15):
    ce = nn.functional.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(SEED)
    model.train()
    for ep in range(epochs):
        order = list(range(len(users)))
        random.shuffle(order)
        tot = nb = 0
        for s in range(0, len(order), bs):
            batch = [users[j] for j in order[s:s + bs]]
            T, H, pad = to_tensors(batch, device)
            mask = random_mask(pad, ratio, gen).to(device)
            lt, lg, la, lc = model(T["types"], T["amounts"], T["gaps"], H,
                                   T["cnov"], pad, mask)
            loss = (ce(lt[mask], T["types"][mask]) +
                    ce(lg[mask], T["gaps"][mask]) +
                    ce(la[mask], T["amounts"][mask]) +
                    ce(lc[mask], T["cnov"][mask]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, users, device, stride=7, bs=64):
    ce = nn.functional.cross_entropy
    model.eval()
    keys = ("t", "g", "a", "c")
    out = [{k: np.zeros(len(u["types"])) for k in keys} for u in users]
    for i, u in enumerate(users):
        out[i]["gb"] = np.array(u["gaps"])
    for r in range(stride):
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            T, H, pad = to_tensors(batch, device)
            pos = torch.arange(T["types"].size(1))
            mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
            if not mask.any():
                continue
            lt, lg, la, lc = model(T["types"], T["amounts"], T["gaps"], H,
                                   T["cnov"], pad, mask.to(device))
            logits = {"t": (lt, "types"), "g": (lg, "gaps"),
                      "a": (la, "amounts"), "c": (lc, "cnov")}
            for i in range(len(batch)):
                idx = mask[i].nonzero().flatten()
                if not len(idx):
                    continue
                for k, (lg_, f) in logits.items():
                    out[s+i][k][idx.numpy()] = ce(lg_[i, idx], T[f][i, idx],
                                                  reduction="none").cpu().numpy()
    return out


def bucket_norm(pcs_tr, pcs, key):
    av = np.concatenate([p[key] for p in pcs_tr])
    ab = np.concatenate([p["gb"] for p in pcs_tr])
    mu = np.full(N_GAP, av.mean()); sd = np.full(N_GAP, max(av.std(), 1e-3))
    for b in range(N_GAP):
        m = ab == b
        if m.sum() >= 30:
            mu[b], sd[b] = av[m].mean(), max(av[m].std(), 1e-3)
    return [(p[key] - mu[np.clip(p["gb"], 0, N_GAP-1)]) /
            sd[np.clip(p["gb"], 0, N_GAP-1)] for p in pcs]


def topk_mean(a, k):
    return float(np.sort(a)[-min(k, len(a)):].mean())


def report(name, y, s):
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    print(f"  {name:<28} AUC={auc:.4f}  KS={np.max(tpr-fpr):.4f}  "
          f"R@FPR1%={r1:.1%}", flush=True)
    return auc


def main():
    device = "cpu"
    users = load()
    normal = [u for u in users if u["label"] == 0]
    phish = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(normal)
    enc = build_encoder(normal[:4000])
    train_u = [enc(u) for u in normal[:4000]]
    test_u = [enc(u) for u in normal[4000:5000]] + [enc(u) for u in phish]
    y = np.array([u["label"] for u in test_u])
    print(f"训练 {len(train_u)} | 测试 {int((y==0).sum())}白+{int(y.sum())}黑",
          flush=True)
    # 快速摸底: cnov 首次占比的裸区分度
    fr = [np.mean(np.array(u["cnov"]) == 1) for u in test_u]
    print(f"裸特征摸底: 首次对手方占比 AUC = "
          f"{roc_auc_score(y, np.array(fr)):.4f}\n", flush=True)

    print("== MEM4C 训练 (+cnov 字段与预测头) ==", flush=True)
    model = MEM4C().to(device)
    train(model, train_u, device)
    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, train_u, device)
    pcs = per_position_ce(model, test_u, device)
    z = {k: bucket_norm(pcs_tr, pcs, k) for k in ("t", "g", "a", "c")}
    combos = {"仅amt": z["a"], "仅cnov": z["c"], "仅type": z["t"],
              "amt+cnov": [a+b for a, b in zip(z["a"], z["c"])],
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
    print(f"\nv1(无对手方) = 0.9035 | v3(+cnov) 最优 = {best[0]:.4f} "
          f"({best[1]})", flush=True)


if __name__ == "__main__":
    main()
