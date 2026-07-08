# -*- coding: utf-8 -*-
"""思想4·Contrastive(CoLES-lite): 同一用户两个随机子片段=正对, 批内他人=负对,
InfoNCE 训练专用表征。检验: 对比表征聚类是否优于 MEM 副产品嵌入;
消融: 片段长度增强(20-40) vs 固定长度。检测=5近白距离。"""
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import bp_common as C
import mem_rich as M

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
d = C.load_art()
users = M.load("data_cluster.jsonl", gq=M.global_quantiles("data_cluster.jsonl"))
uid2i = {u["uid"]: i for i, u in enumerate(users)}
meta = json.load(open("blueprint_meta.json"))
users = [users[uid2i[u]] for u in meta["uid"]]
tr_users = [u for u, t in zip(users, d["is_train"]) if t]
print("== 思想4 Contrastive(CoLES-lite) ==", flush=True)

FIELDS = ["type", "gamt", "gap", "hour", "res", "ch", "ip", "pamt"]
LISTF = ("type", "res", "ch", "ip", "gamt", "pamt", "gap", "hour")


def slice_u(u, rng, lo=20, hi=40):
    n = len(u["type"])
    L = min(rng.randint(lo, hi), n)
    a = rng.randint(0, n - L)
    return {f: u[f][a:a + L] for f in LISTF}


def forward_pool(model, batch, device="cpu"):
    T, pad = M.to_tensors(batch, device)
    parts = []
    for f in model.fields:
        if f == "hour":
            h = T["hour"]
            parts.append(model.hour_proj(torch.stack(
                [torch.sin(2 * math.pi * h / 24),
                 torch.cos(2 * math.pi * h / 24)], -1)))
        else:
            parts.append(model.embs["e_" + f](T[f]))
    x = model.in_proj(torch.cat(parts, -1))
    x = x + model.pos_emb(torch.arange(x.size(1)))[None]
    h = model.encoder(x, src_key_padding_mask=pad)
    m = (~pad).float().unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1)


def train_coles(lo, hi, epochs=15, bs=32, temp=0.2):
    torch.manual_seed(SEED)
    model = M.MEMRich(FIELDS)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    rng = random.Random(SEED)
    for ep in range(epochs):
        order = list(range(len(tr_users)))
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [tr_users[j] for j in order[s:s + bs]]
            if len(batch) < 4:
                continue
            views = [slice_u(u, rng, lo, hi) for u in batch] + \
                    [slice_u(u, rng, lo, hi) for u in batch]
            z = F.normalize(forward_pool(model, views), dim=1)
            sim = z @ z.T / temp
            sim.fill_diagonal_(-1e9)
            B = len(batch)
            tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)])
            loss = F.cross_entropy(sim, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1:2d} loss {tot/nb:.4f}", flush=True)
    return model


@torch.no_grad()
def all_emb(model):
    model.eval()
    out = []
    for s in range(0, len(users), 64):
        out.append(forward_pool(model, users[s:s + 64]).numpy())
    return np.concatenate(out)


def evaluate(tag, E):
    sc = StandardScaler().fit(E[d["is_train"]])
    Es = sc.transform(E)
    nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(E[d["is_train"]]))
    s = nn.kneighbors(Es)[0][:, -1]
    C.row(tag, C.det_auc(d, s), C.white_ari(d, Es), C.black_ari(d, Es))
    return s


model = train_coles(20, 40)
s = evaluate("对比表征(片段20-40)", all_emb(model))
C.save_score("e4_coles", s)
model2 = train_coles(30, 30)
evaluate("消融: 固定片段30", all_emb(model2))
C.row("参照: MEM 嵌入", None, C.white_ari(d, d["emb"]), C.black_ari(d, d["emb"]))
