# -*- coding: utf-8 -*-
"""
MEM (Masked Event Model) 异常检测实验
--------------------------------------
方案: 只用白样本训练一个双向 Transformer, 随机 mask 15% 事件,
同时预测被 mask 事件的 [类型] 和 [与前一事件的时间间隔分桶]。
异常分数 = 测试序列被 mask 位置的平均交叉熵(多轮 mask 取平均)。
期望: 黑样本"敏感操作密集爆发"违背白样本的 (类型,间隔) 联合规律,
重建误差显著更高。

对照基线(均为使用标签的 oracle, 用于验证数据设计/量化信号泄漏):
  B1 词频向量 + LR      —— 应≈0.5, 证明词频无信号
  B2 间隔直方图 + LR    —— 量化间隔边缘分布泄漏
  B3 规则: 敏感事件前间隔≤30min 占比 —— 手工特征天花板
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EVENT_TYPES = ["登录", "查余额", "改限额", "改密码", "绑卡", "解绑卡",
               "设备变更", "转入", "转出", "消费", "还款", "借款"]
TYPE2ID = {e: i for i, e in enumerate(EVENT_TYPES)}
SENSITIVE = {"改限额", "改密码", "绑卡", "解绑卡", "设备变更"}
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]  # 8 个真实桶
GAP_BOS = len(GAP_EDGES) + 1                              # 首事件专用桶 → 共 9 类
N_GAP = GAP_BOS + 1
MAX_LEN = 200

def gap_bucket(sec):
    return sum(sec > e for e in GAP_EDGES)

def amount_bucket(ev):
    a = ev.get("amount")
    if a is None:
        return 0
    return 1 if a < 5000 else (2 if a < 50000 else 3)

def load(path):
    users = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            types, amounts, gaps, hours = [], [], [], []
            prev = None
            for ev in r["events"][:MAX_LEN]:
                t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
                types.append(TYPE2ID[ev["type"]])
                amounts.append(amount_bucket(ev))
                gaps.append(GAP_BOS if prev is None
                            else gap_bucket((t - prev).total_seconds()))
                hours.append(t.hour + t.minute / 60)
                prev = t
            users.append({"uid": r["user_id"], "label": r["label"],
                          "sub": r["sub_label"],
                          "types": types, "amounts": amounts,
                          "gaps": gaps, "hours": hours})
    return users

def pad_batch(batch, device):
    n = len(batch)
    L = max(len(u["types"]) for u in batch)
    tp = torch.zeros(n, L, dtype=torch.long)
    am = torch.zeros(n, L, dtype=torch.long)
    gp = torch.zeros(n, L, dtype=torch.long)
    hr = torch.zeros(n, L, 2)
    pad = torch.ones(n, L, dtype=torch.bool)  # True = padding
    for i, u in enumerate(batch):
        m = len(u["types"])
        tp[i, :m] = torch.tensor(u["types"])
        am[i, :m] = torch.tensor(u["amounts"])
        gp[i, :m] = torch.tensor(u["gaps"])
        h = torch.tensor(u["hours"])
        hr[i, :m, 0] = torch.sin(2 * math.pi * h / 24)
        hr[i, :m, 1] = torch.cos(2 * math.pi * h / 24)
        pad[i, :m] = False
    return tp.to(device), am.to(device), gp.to(device), hr.to(device), pad.to(device)


class MEM(nn.Module):
    def __init__(self, d=64, n_types=len(EVENT_TYPES), n_amount=4,
                 n_gap=N_GAP, max_len=MAX_LEN):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, 32)
        self.amount_emb = nn.Embedding(n_amount, 8)
        self.gap_emb = nn.Embedding(n_gap, 16)
        self.hour_proj = nn.Linear(2, 8)
        self.in_proj = nn.Linear(32 + 8 + 16 + 8, d)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_type = nn.Linear(d, n_types)
        self.head_gap = nn.Linear(d, n_gap)

    def encode(self, tp, am, gp, hr, pad, mask=None):
        x = self.in_proj(torch.cat([self.type_emb(tp), self.amount_emb(am),
                                    self.gap_emb(gp), self.hour_proj(hr)], -1))
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_emb.expand_as(x), x)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)[None]
        return self.encoder(x, src_key_padding_mask=pad)

    def forward(self, tp, am, gp, hr, pad, mask):
        h = self.encode(tp, am, gp, hr, pad, mask)
        return self.head_type(h), self.head_gap(h)


def random_mask(pad, ratio, gen):
    """在非 padding 位置随机 mask ratio 比例(每条序列至少 1 个)"""
    scores = torch.rand(pad.shape, generator=gen)
    scores[pad] = -1.0
    mask = scores > (1 - ratio)
    for i in range(pad.size(0)):
        if not mask[i].any():
            valid = (~pad[i]).nonzero().flatten()
            mask[i, valid[torch.randint(len(valid), (1,), generator=gen)]] = True
    return mask


def masked_ce(logits_t, logits_g, tp, gp, mask):
    """返回 (type_ce_sum, gap_ce_sum, count), 逐样本"""
    ce = nn.functional.cross_entropy
    n = tp.size(0)
    t_sum = torch.zeros(n)
    g_sum = torch.zeros(n)
    cnt = mask.sum(1).float()
    for i in range(n):
        m = mask[i]
        if m.any():
            t_sum[i] = ce(logits_t[i, m], tp[i, m], reduction="sum").cpu()
            g_sum[i] = ce(logits_g[i, m], gp[i, m], reduction="sum").cpu()
    return t_sum, g_sum, cnt.cpu()


def train(model, whites, device, epochs=40, bs=32, lr=1e-3, ratio=0.15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(SEED)
    model.train()
    for ep in range(epochs):
        order = list(range(len(whites)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [whites[j] for j in order[s:s + bs]]
            tp, am, gp, hr, pad = pad_batch(batch, device)
            mask = random_mask(pad, ratio, gen).to(device)
            lt, lg = model(tp, am, gp, hr, pad, mask)
            m = mask
            loss = (nn.functional.cross_entropy(lt[m], tp[m]) +
                    nn.functional.cross_entropy(lg[m], gp[m]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}")


@torch.no_grad()
def score(model, users, device, rounds=10, bs=64, ratio=0.15):
    model.eval()
    t_all = np.zeros(len(users))
    g_all = np.zeros(len(users))
    c_all = np.zeros(len(users))
    for r in range(rounds):
        gen = torch.Generator().manual_seed(1000 + r)
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            tp, am, gp, hr, pad = pad_batch(batch, device)
            mask = random_mask(pad, ratio, gen).to(device)
            lt, lg = model(tp, am, gp, hr, pad, mask)
            ts, gs, cs = masked_ce(lt, lg, tp, gp, mask)
            t_all[s:s + len(batch)] += ts.numpy()
            g_all[s:s + len(batch)] += gs.numpy()
            c_all[s:s + len(batch)] += cs.numpy()
    return t_all / c_all, g_all / c_all  # 每 mask 位置平均 CE


def ks_stat(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def report(name, y, s):
    auc = roc_auc_score(y, s)
    print(f"  {name:<32} AUC={auc:.4f}  KS={ks_stat(y, s):.4f}")
    return auc


def main():
    device = "cpu"
    users = load("data_temporal.jsonl")
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_w = whites[:400], whites[400:]
    test_users = test_w + blacks
    y = np.array([u["label"] for u in test_users])
    print(f"训练集: {len(train_w)} 白 | 测试集: {len(test_w)} 白 + {len(blacks)} 黑\n")

    # ---------- Oracle 基线(使用标签, 仅用于验证数据/量化泄漏) ----------
    print("== Oracle 基线 (5折交叉验证, 全量 1000 用户) ==")
    y_all = np.array([u["label"] for u in users])
    freq = np.array([[u["types"].count(k) for k in range(len(EVENT_TYPES))]
                     for u in users], dtype=float)
    freq /= freq.sum(1, keepdims=True)
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p1 = cross_val_predict(lr_pipe, freq, y_all, cv=5,
                           method="predict_proba")[:, 1]
    report("B1 词频向量+LR(应≈0.5)", y_all, p1)

    gaph = np.zeros((len(users), N_GAP))
    for i, u in enumerate(users):
        for g in u["gaps"]:
            gaph[i, g] += 1
        gaph[i] /= max(1, len(u["gaps"]))
    p2 = cross_val_predict(lr_pipe, gaph, y_all, cv=5,
                           method="predict_proba")[:, 1]
    report("B2 间隔直方图+LR(边缘泄漏)", y_all, p2)

    rule = np.array([
        (sum(1 for k, tpid in enumerate(u["types"])
             if EVENT_TYPES[tpid] in SENSITIVE and u["gaps"][k] <= 2) /
         max(1, sum(1 for tpid in u["types"] if EVENT_TYPES[tpid] in SENSITIVE)))
        for u in users])
    report("B3 规则:敏感事件前间隔≤30min占比", y_all, rule)

    # ---------- MEM ----------
    print("\n== MEM 训练 (仅 400 白样本, 双目标: 类型+间隔桶) ==")
    model = MEM().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_param/1e3:.1f}K")
    train(model, train_w, device)

    print("\n== MEM 异常分数 (10 轮 mask 平均) ==")
    s_type, s_gap = score(model, test_users, device)
    s_sum = s_type + s_gap
    report("MEM 总分(type+gap CE)", y, s_sum)
    report("MEM 仅类型 CE", y, s_type)
    report("MEM 仅间隔 CE", y, s_gap)

    print("\n== 分欺诈模式 AUC (白 vs 单一模式) ==")
    for sub, name in [(1, "盗号"), (2, "套现"), (3, "洗钱")]:
        idx = [i for i, u in enumerate(test_users)
               if u["label"] == 0 or u["sub"] == sub]
        report(f"  {name}", y[idx], s_sum[idx])

    print("\n== 分数分布 ==")
    for nm, arr in [("白(测试)", s_sum[y == 0]), ("黑", s_sum[y == 1])]:
        print(f"  {nm}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"p10={np.percentile(arr,10):.4f}  p90={np.percentile(arr,90):.4f}")

    np.savez("mem_scores.npz", score=s_sum, s_type=s_type, s_gap=s_gap,
             label=y, uid=[u["uid"] for u in test_users])
    print("\n分数已保存 → mem_scores.npz")


if __name__ == "__main__":
    main()
