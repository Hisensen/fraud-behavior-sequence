# -*- coding: utf-8 -*-
"""
实验矩阵·准备脚本 —— 训练一次 MEM, 导出全部思想实验共用的产物:
  emb      (N,64)  行为嵌入(隐层mean-pool)
  fp       (N,F)   惊讶画像(异常指纹)
  stats    (N,~40) 手工聚合统计特征
  mem_topk / mem_mean  MEM 检测分
  y / is_train 标签与划分; blueprint_meta.json 存 wtype/btype
"""
import json
import random

import numpy as np
import torch

import mem_rich as M
from cluster_experiment import embed, surprise_profile

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
PATH = "data_cluster.jsonl"


def stats_feats(u):
    n = len(u["type"])
    tf = [u["type"].count(k) / n for k in range(len(M.EVENT_TYPES))]
    gh = [u["gap"].count(b) / n for b in range(M.N_GAP)]
    ph = [u["pamt"].count(b) / n for b in range(M.N_PAMT)]
    ch = [u["ch"].count(c) / n for c in (1, 2, 3)]
    h = np.array(u["hour"])
    return (tf + gh + ph + ch +
            [float(((h >= 0) & (h < 6)).mean()),
             float(np.mean(np.sin(2 * np.pi * h / 24))),
             float(np.mean(np.cos(2 * np.pi * h / 24))),
             u["res"].count(2) / n, u["ip"].count(2) / n,
             float(np.log(n))])


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
    rng = random.Random(SEED); rng.shuffle(whites)
    train_ids = {u["uid"] for u in whites[:1600]}
    is_train = np.array([u["uid"] in train_ids for u in users])
    y = np.array([u["label"] for u in users])
    train_w = [u for u in users if u["uid"] in train_ids]
    print(f"{len(users)} 用户 | 训练白 {len(train_w)}", flush=True)

    fields = ["type", "gamt", "gap", "hour", "res", "ch", "ip", "pamt"]
    model = M.MEMRich(fields).to(device)
    M.train(model, train_w, device, epochs=30)

    print("导出嵌入/指纹/分数…", flush=True)
    emb = embed(model, users, device)
    pcs_tr = M.per_position_ce(model, train_w, device)
    pcs = M.per_position_ce(model, users, device)
    zs = {k: M.z_norm(pcs_tr, pcs, k) for k in pcs[0] if k != "gb"}
    fp = surprise_profile(users, zs)
    zsum = [sum(v) for v in zip(*zs.values())]
    mem_topk = np.array([M.topk_mean(a, 5) for a in zsum])
    mem_mean = np.array([float(a.mean()) for a in zsum])
    stats = np.array([stats_feats(u) for u in users])

    np.savez("blueprint_art.npz", emb=emb, fp=fp, stats=stats,
             mem_topk=mem_topk, mem_mean=mem_mean, y=y, is_train=is_train)
    json.dump({"uid": [u["uid"] for u in users],
               "wtype": [u["wtype"] for u in users],
               "btype": [u["btype"] for u in users]},
              open("blueprint_meta.json", "w"), ensure_ascii=False)
    torch.save(model.state_dict(), "blueprint_mem.pt")

    from sklearn.metrics import roc_auc_score
    te = ~is_train
    print(f"MEM 检测 AUC(top5)={roc_auc_score(y[te], mem_topk[te]):.4f} "
          f"(mean)={roc_auc_score(y[te], mem_mean[te]):.4f}", flush=True)
    print("prep 完成", flush=True)


if __name__ == "__main__":
    main()
