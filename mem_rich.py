# -*- coding: utf-8 -*-
"""
富字段编码消融 —— 在 data_rich.jsonl 上回答"编码该加什么"。

四档编码(累进):
  E0 现行简单编码: type + 全局金额桶 + 间隔桶 + 时刻
  E1 = E0 + result + channel + ip_change      (状态/渠道字段)
  E2 = E0 + 个人相对金额桶                      (个人基线)
  E3 = 全量
统一 MEM 目标: 整事件遮罩, 多头预测(type+gap 恒有; res/pamt 字段激活时加头)。
打分: 逐位置 CE → 按间隔桶 z-norm → top-k。
Oracle 组(用标签)先量化各信号层强度, 再看无监督各档能吃到多少。
"""
import json
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EVENT_TYPES = ["登录", "查余额", "改限额", "改密码", "绑卡", "解绑卡",
               "设备变更", "转入", "转出", "消费", "还款", "借款"]
T2I = {e: i for i, e in enumerate(EVENT_TYPES)}
SENSITIVE = {"改限额", "改密码", "绑卡", "解绑卡", "设备变更"}
CH2I = {"APP": 1, "WEB": 2, "POS": 3}
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_BOS = len(GAP_EDGES) + 1
N_GAP = GAP_BOS + 1
RATIO_EDGES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]     # 个人比值桶
N_PAMT = len(RATIO_EDGES) + 2                      # 0=N/A
N_GAMT = 8
MAX_LEN = 150

FIELD_DIMS = {"type": 32, "res": 4, "ch": 8, "ip": 4,
              "gamt": 8, "pamt": 16, "gap": 16, "hour": 8}
VARIANTS = {
    "E0 简单(type+全局金额+gap+hour)": ["type", "gamt", "gap", "hour"],
    "E1 +result/channel/ip": ["type", "gamt", "gap", "hour", "res", "ch", "ip"],
    "E2 +个人相对金额桶": ["type", "gamt", "gap", "hour", "pamt"],
    "E3 全量": ["type", "gamt", "gap", "hour", "res", "ch", "ip", "pamt"],
}


def load(path="data_rich.jsonl", gq=None):
    users, all_amts = [], []
    raw = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in raw:
        for ev in r["events"]:
            if "amount" in ev:
                all_amts.append(ev["amount"])
    users_out = []
    for r in raw:
        evs = r["events"][:MAX_LEN]
        amts = [ev.get("amount") for ev in evs]
        nz = [a for a in amts if a]
        med = np.median(nz) if nz else 1.0
        types, res, ch, ip, gamt, pamt, gaps, hours = [], [], [], [], [], [], [], []
        tss = []
        prev = None
        for ev in evs:
            t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
            tss.append(t.timestamp())
            types.append(T2I[ev["type"]])
            res.append({"成功": 1, "失败": 2}.get(ev.get("result"), 0))
            ch.append(CH2I.get(ev.get("channel"), 0))
            ip.append(0 if "ip_change" not in ev else 1 + ev["ip_change"])
            a = ev.get("amount")
            gamt.append(0 if not a else 1 + int(np.searchsorted(gq, a)))
            pamt.append(0 if not a else
                        1 + int(np.searchsorted(RATIO_EDGES, a / med)))
            gaps.append(GAP_BOS if prev is None else
                        int(np.searchsorted(GAP_EDGES,
                                            max((t - prev).total_seconds(), 0),
                                            side="left")))
            hours.append(t.hour + t.minute / 60)
            prev = t
        users_out.append({"uid": r["user_id"], "label": r["label"],
                          "type": types, "res": res, "ch": ch, "ip": ip,
                          "gamt": gamt, "pamt": pamt, "gap": gaps,
                          "hour": hours, "ts": tss})
    return users_out


def global_quantiles(path="data_rich.jsonl"):
    amts = []
    for l in open(path, encoding="utf-8"):
        for ev in json.loads(l)["events"]:
            if "amount" in ev:
                amts.append(ev["amount"])
    return np.quantile(np.array(amts), np.linspace(0, 1, N_GAMT)[1:-1])


class MEMRich(nn.Module):
    N_CLS = {"type": len(EVENT_TYPES), "res": 3, "ch": 4, "ip": 3,
             "gamt": N_GAMT, "pamt": N_PAMT, "gap": N_GAP}

    def __init__(self, fields, d=64):
        super().__init__()
        self.fields = fields
        self.embs = nn.ModuleDict()
        in_dim = 0
        for f in fields:
            if f == "hour":
                self.hour_proj = nn.Linear(2, FIELD_DIMS["hour"])
            else:
                # 键加前缀, 避开 nn.Module 保留属性名(type 等)
                self.embs["e_" + f] = nn.Embedding(self.N_CLS[f], FIELD_DIMS[f])
            in_dim += FIELD_DIMS[f]
        self.in_proj = nn.Linear(in_dim, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.pos_emb = nn.Embedding(MAX_LEN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.heads = nn.ModuleDict({"h_type": nn.Linear(d, self.N_CLS["type"]),
                                    "h_gap": nn.Linear(d, N_GAP)})
        if "res" in fields:
            self.heads["h_res"] = nn.Linear(d, 3)
        if "pamt" in fields:
            self.heads["h_pamt"] = nn.Linear(d, N_PAMT)

    def head_targets(self):
        m = {"h_type": "type", "h_gap": "gap"}
        if "h_res" in self.heads: m["h_res"] = "res"
        if "h_pamt" in self.heads: m["h_pamt"] = "pamt"
        return m

    def forward(self, T, pad, mask):
        parts = []
        for f in self.fields:
            if f == "hour":
                h = T["hour"]
                parts.append(self.hour_proj(torch.stack(
                    [torch.sin(2 * math.pi * h / 24),
                     torch.cos(2 * math.pi * h / 24)], -1)))
            else:
                parts.append(self.embs["e_" + f](T[f]))
        x = self.in_proj(torch.cat(parts, -1))
        x = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        h = self.encoder(x, src_key_padding_mask=pad)
        return {k: head(h) for k, head in self.heads.items()}


def to_tensors(batch, device):
    n = len(batch)
    L = max(len(u["type"]) for u in batch)
    T = {}
    for f in ("type", "res", "ch", "ip", "gamt", "pamt", "gap"):
        M = torch.zeros(n, L, dtype=torch.long)
        for i, u in enumerate(batch):
            M[i, :len(u[f])] = torch.tensor(u[f])
        T[f] = M.to(device)
    H = torch.zeros(n, L)
    pad = torch.ones(n, L, dtype=torch.bool)
    for i, u in enumerate(batch):
        H[i, :len(u["hour"])] = torch.tensor(u["hour"])
        pad[i, :len(u["hour"])] = False
    T["hour"] = H.to(device)
    return T, pad.to(device)


def random_mask(pad, ratio, gen):
    sc = torch.rand(pad.shape, generator=gen)
    sc[pad] = -1.0
    mask = sc > (1 - ratio)
    for i in range(pad.size(0)):
        if not mask[i].any():
            v = (~pad[i]).nonzero().flatten()
            mask[i, v[torch.randint(len(v), (1,), generator=gen)]] = True
    return mask


def train(model, users, device, epochs=30, bs=32, lr=1e-3, ratio=0.15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.functional.cross_entropy
    gen = torch.Generator().manual_seed(SEED)
    hm = model.head_targets()
    model.train()
    for ep in range(epochs):
        order = list(range(len(users)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [users[j] for j in order[s:s + bs]]
            T, pad = to_tensors(batch, device)
            mask = random_mask(pad, ratio, gen).to(device)
            out = model(T, pad, mask)
            loss = sum(ce(out[k][mask], T[f][mask]) for k, f in hm.items())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, users, device, stride=7, bs=64):
    model.eval()
    ce = nn.functional.cross_entropy
    hm = model.head_targets()
    out = [{k: np.zeros(len(u["type"])) for k in hm} for u in users]
    for i, u in enumerate(users):
        out[i]["gb"] = np.array(u["gap"])
    for r in range(stride):
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            T, pad = to_tensors(batch, device)
            pos = torch.arange(T["type"].size(1))
            mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
            if not mask.any():
                continue
            o = model(T, pad, mask.to(device))
            for i in range(len(batch)):
                idx = mask[i].nonzero().flatten()
                if len(idx) == 0:
                    continue
                for k, f in hm.items():
                    out[s+i][k][idx.numpy()] = ce(
                        o[k][i, idx], T[f][i, idx],
                        reduction="none").cpu().numpy()
    return out


def z_norm(pcs_tr, pcs, key):
    av = np.concatenate([p[key] for p in pcs_tr])
    ab = np.concatenate([p["gb"] for p in pcs_tr])
    mu = np.full(N_GAP, av.mean()); sd = np.full(N_GAP, max(av.std(), 1e-3))
    for b in range(N_GAP):
        m = ab == b
        if m.sum() >= 30:
            mu[b], sd[b] = av[m].mean(), max(av[m].std(), 1e-3)
    return [(p[key] - mu[np.clip(p["gb"], 0, N_GAP-1)]) /
            sd[np.clip(p["gb"], 0, N_GAP-1)] for p in pcs]


def topk_mean(a, k=5):
    return float(np.sort(a)[-min(k, len(a)):].mean())


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return (roc_auc_score(y, s), float(np.max(tpr - fpr)),
            float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]))


def main():
    device = "cpu"
    gq = global_quantiles()
    users = load(gq=gq)
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_w = whites[:1000], whites[1000:]
    test_u = test_w + blacks
    y = np.array([u["label"] for u in test_u])
    print(f"训练 {len(train_w)} 白 | 测试 {len(test_w)} 白 + {len(blacks)} 黑\n",
          flush=True)

    # ---------- Oracle 组: 各信号层强度 ----------
    print("== Oracle 信号层摸底 (5折CV, 用标签) ==", flush=True)
    allu = train_w + test_u
    ya = np.array([u["label"] for u in allu])
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    def cv(name, X):
        p = cross_val_predict(lr_pipe, np.array(X), ya, cv=5,
                              method="predict_proba")[:, 1]
        auc, ks, r1 = metrics(ya, p)
        print(f"  {name:<30} AUC={auc:.4f}  R@FPR1%={r1:.1%}", flush=True)

    cv("O1 词频", [[u["type"].count(k) / len(u["type"])
                    for k in range(len(EVENT_TYPES))] for u in allu])
    cv("O2 全局金额统计", [[np.mean([g for g in u["gamt"] if g]) if any(u["gamt"]) else 0,
                            max(u["gamt"]), np.mean(u["gamt"])] for u in allu])
    cv("O3 个人比值统计", [[max(u["pamt"]), np.mean([p for p in u["pamt"] if p] or [0]),
                            sum(1 for p in u["pamt"] if p == N_PAMT - 1)]
                           for u in allu])
    cv("O4 result/渠道/ip统计",
       [[sum(1 for r in u["res"] if r == 2),
         sum(1 for i_ in u["ip"] if i_ == 2),
         (lambda c: 1 - c.count(max(set(c), key=c.count)) / len(c) if c else 0)
         ([c for c in u["ch"] if c])] for u in allu])
    cv("O5 时序密集统计",
       [[sum(1 for k in range(len(u["type"]))
             if EVENT_TYPES[u["type"][k]] in SENSITIVE and u["gap"][k] <= 1),
         sum(1 for g in u["gap"] if g == 0)] for u in allu])

    # ---------- 编码消融 ----------
    results = {}
    for name, fields in VARIANTS.items():
        print(f"\n== {name} | 字段: {','.join(fields)} ==", flush=True)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        model = MEMRich(fields).to(device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"    参数 {n_par/1e3:.0f}K", flush=True)
        train(model, train_w, device)
        pcs_tr = per_position_ce(model, train_w, device)
        pcs = per_position_ce(model, test_u, device)
        zs = {k: z_norm(pcs_tr, pcs, k) for k in pcs[0] if k != "gb"}
        best = (0, None)
        # 单头 + 全头求和, top-k 网格
        combos = {k: v for k, v in zs.items()}
        combos["SUM"] = [sum(vals) for vals in zip(*zs.values())]
        for cn, z in combos.items():
            for k in (3, 5, 10):
                s = np.array([topk_mean(a, k) for a in z])
                auc, ks, r1 = metrics(y, s)
                if auc > best[0]:
                    best = (auc, f"{cn} top-{k}", ks, r1)
        auc, cfg, ks, r1 = best
        results[name] = best
        print(f"    最优: {cfg:<14} AUC={auc:.4f}  KS={ks:.4f}  "
              f"R@FPR1%={r1:.1%}", flush=True)

    print("\n" + "=" * 62, flush=True)
    print("== 编码消融总表 ==", flush=True)
    for name, (auc, cfg, ks, r1) in results.items():
        print(f"  {name:<28} AUC={auc:.4f}  R@FPR1%={r1:>6.1%}  ({cfg})",
              flush=True)


if __name__ == "__main__":
    main()
