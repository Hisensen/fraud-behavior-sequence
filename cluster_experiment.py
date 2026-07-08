# -*- coding: utf-8 -*-
"""
聚类可行性实验 —— 从"二分类打分"升级到"人群/手法结构发现"。

核心假设(要验证的设计律):
  白样本聚类 = "他是谁"  → 用模型隐层的序列嵌入(mean-pool), 聚出人群原型
  黑样本聚类 = "他哪里不对劲" → 用惊讶画像向量(各头z分统计+可疑位置类型分布),
                聚出作案手法
  交叉配对应当失败/退化: 嵌入聚黑样本只会聚出他们的底座人群,
                惊讶画像聚白样本只是噪声。

数据: data_cluster.jsonl (4 白人群原型 × 4 黑手法, 均有真值标签)
模型: 复用 mem_rich.MEMRich E3 全量编码, 只用白样本训练。
评估: ARI / NMI vs 真值; 轮廓系数选 k; 词频向量作 baseline。
"""
import json
import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             roc_auc_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

import mem_rich as M

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
PATH = "data_cluster.jsonl"
EVENT_TYPES = M.EVENT_TYPES


@torch.no_grad()
def embed(model, users, device, bs=64):
    """mean-pool 编码器隐层 → 每用户一个 d 维行为嵌入。"""
    import math
    model.eval()
    out = []
    for s in range(0, len(users), bs):
        batch = users[s:s + bs]
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
        x = x + model.pos_emb(torch.arange(x.size(1), device=device))[None]
        h = model.encoder(x, src_key_padding_mask=pad)
        m = (~pad).float().unsqueeze(-1)
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)


def surprise_profile(users, zs_by_head):
    """惊讶画像: 每头 z 分统计(哪个头惊讶) + 高惊讶位置的事件类型分布(惊讶在什么事上)。"""
    heads = sorted(zs_by_head)
    feats = []
    for i, u in enumerate(users):
        row = []
        zsum = None
        for h in heads:
            z = zs_by_head[h][i]
            row += [float(z.mean()), M.topk_mean(z, 5), float(z.max()),
                    float((z > 2).mean())]
            zsum = z if zsum is None else zsum + z
        w = np.clip(zsum, 0, None)
        hist = np.zeros(len(EVENT_TYPES))
        for k, t in enumerate(u["type"]):
            hist[t] += w[k]
        hist = hist / max(hist.sum(), 1e-6)
        # 高惊讶位置的间隔桶分布(手法的节奏指纹)
        top = np.argsort(-zsum)[:10]
        ghist = np.zeros(M.N_GAP)
        for k in top:
            ghist[min(u["gap"][k], M.N_GAP - 1)] += 1
        ghist /= len(top)
        feats.append(row + hist.tolist() + ghist.tolist())
    return np.array(feats)


def freq_vector(users):
    return np.array([[u["type"].count(k) / len(u["type"])
                      for k in range(len(EVENT_TYPES))] for u in users])


def cluster_eval(name, X, y_true, k_true):
    Xs = StandardScaler().fit_transform(X)
    # 轮廓系数选 k
    sils = {}
    for k in range(2, 9):
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs)
        sils[k] = silhouette_score(Xs, lab)
    k_pick = max(sils, key=sils.get)
    lab = KMeans(k_true, n_init=10, random_state=SEED).fit_predict(Xs)
    ari = adjusted_rand_score(y_true, lab)
    nmi = normalized_mutual_info_score(y_true, lab)
    print(f"  {name:<34} ARI={ari:.3f}  NMI={nmi:.3f}  "
          f"轮廓选k={k_pick}(真值{k_true})")
    return lab, ari


def confusion(lab, names_true, title):
    cats = sorted(set(names_true))
    ks = sorted(set(lab))
    print(f"    {title}: 行=真值, 列=簇")
    print("    " + " " * 10 + "".join(f"簇{k:<4}" for k in ks))
    for c in cats:
        row = [sum(1 for l, n in zip(lab, names_true) if n == c and l == k)
               for k in ks]
        print(f"    {c:<10}" + "".join(f"{v:<5}" for v in row))


def main():
    device = "cpu"
    gq = M.global_quantiles(PATH)
    users = M.load(PATH, gq=gq)
    meta = {json.loads(l)["user_id"]: json.loads(l)
            for l in open(PATH, encoding="utf-8")}
    for u in users:
        u["wtype"] = meta[u["uid"]]["wtype"]
        u["btype"] = meta[u["uid"]]["btype"]

    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_w = whites[:1600], whites[1600:]
    print(f"训练 {len(train_w)} 白 | 测试 {len(test_w)} 白 + {len(blacks)} 黑\n",
          flush=True)

    fields = ["type", "gamt", "gap", "hour", "res", "ch", "ip", "pamt"]
    model = M.MEMRich(fields).to(device)
    M.train(model, train_w, device, epochs=30)

    # ---------- 先确认检测本身没垮 ----------
    test_u = test_w + blacks
    y = np.array([u["label"] for u in test_u])
    pcs_tr = M.per_position_ce(model, train_w, device)
    pcs = M.per_position_ce(model, test_u, device)
    zs = {k: M.z_norm(pcs_tr, pcs, k) for k in pcs[0] if k != "gb"}
    zsum = [sum(v) for v in zip(*zs.values())]
    s = np.array([M.topk_mean(a, 5) for a in zsum])
    print(f"\n检测 sanity check: SUM top-5 AUC = {roc_auc_score(y, s):.4f}\n",
          flush=True)

    # ---------- 任务A: 白样本人群聚类 ----------
    print("== 任务A: 白样本聚出人群原型 (真值: 4 种) ==", flush=True)
    wa = whites  # 全部白样本
    y_w = [u["wtype"] for u in wa]
    emb_w = embed(model, wa, device)
    lab, _ = cluster_eval("行为嵌入(隐层mean-pool)", emb_w, y_w, 4)
    confusion(lab, y_w, "嵌入聚类混淆")
    cluster_eval("词频向量 baseline", freq_vector(wa), y_w, 4)
    pcs_w = M.per_position_ce(model, wa, device)
    zs_w = {k: M.z_norm(pcs_tr, pcs_w, k) for k in pcs_w[0] if k != "gb"}
    cluster_eval("惊讶画像(交叉配对,应退化)", surprise_profile(wa, zs_w), y_w, 4)

    # ---------- 任务B: 黑样本手法聚类 ----------
    print("\n== 任务B: 黑样本聚出作案手法 (真值: 4 种) ==", flush=True)
    y_b = [u["btype"] for u in blacks]
    y_b_base = [u["wtype"] for u in blacks]
    pcs_b = M.per_position_ce(model, blacks, device)
    zs_b = {k: M.z_norm(pcs_tr, pcs_b, k) for k in pcs_b[0] if k != "gb"}
    prof_b = surprise_profile(blacks, zs_b)
    lab, _ = cluster_eval("惊讶画像(各头z统计+可疑位置分布)", prof_b, y_b, 4)
    confusion(lab, y_b, "惊讶画像聚类混淆")
    emb_b = embed(model, blacks, device)
    lab2, _ = cluster_eval("行为嵌入(交叉配对)", emb_b, y_b, 4)
    ari_base = adjusted_rand_score(y_b_base, lab2)
    print(f"    ↑ 嵌入聚黑样本 vs 手法真值如上; vs 底座人群真值 "
          f"ARI={ari_base:.3f} (预期: 嵌入认人不认手法)")
    cluster_eval("词频向量 baseline", freq_vector(blacks), y_b, 4)


if __name__ == "__main__":
    main()
