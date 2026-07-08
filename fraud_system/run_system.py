# -*- coding: utf-8 -*-
"""
一体化反欺诈系统 —— 7 个验证有效方法的完整流水线, 一条命令跑通:
  ① MEM 惊讶度(mean/top5)   ② 统计特征+iForest   ③ 原型距离(K=16)
  ④a 第5近白距离  ④b 案件库检索归因  ⑤ 嵌入聚人群  ⑥ 指纹聚手法  ⑦ 秩融合
输出 system_result.json, 由 make_report.py 渲染成 report.html。
自包含: 只依赖本文件夹内的 mem_rich.py / cluster_experiment.py / data_cluster.jsonl。
"""
import json
import random

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import mem_rich as M
from cluster_experiment import embed, surprise_profile

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
PATH = "data_cluster.jsonl"
R = {}   # 最终结果


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
             u["res"].count(2) / n, u["ip"].count(2) / n, float(np.log(n))])


def auc_r1(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return (float(roc_auc_score(y, s)),
            float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]))


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
    is_tr = np.array([u["uid"] in train_ids for u in users])
    te = ~is_tr
    y = np.array([u["label"] for u in users])
    yte = y[te]
    wtype = np.array([u["wtype"] for u in users])
    btype = np.array([u["btype"] or "" for u in users])
    train_w = [u for u in users if u["uid"] in train_ids]
    R["setup"] = {"n_users": len(users), "n_train_w": len(train_w),
                  "n_test_w": int((te & (y == 0)).sum()),
                  "n_black": int(y.sum())}
    print(f"数据 {len(users)} | 训练白 {len(train_w)} | "
          f"测试 {R['setup']['n_test_w']}白+{R['setup']['n_black']}黑", flush=True)

    # ========== 方法① MEM ==========
    print("① 训练 MEM (只用白样本, 30 epochs)…", flush=True)
    fields = ["type", "gamt", "gap", "hour", "res", "ch", "ip", "pamt"]
    model = M.MEMRich(fields).to(device)
    M.train(model, train_w, device, epochs=30)
    print("① 逐位置惊讶度 + 归一化…", flush=True)
    pcs_tr = M.per_position_ce(model, train_w, device)
    pcs = M.per_position_ce(model, users, device)
    zs = {k: M.z_norm(pcs_tr, pcs, k) for k in pcs[0] if k != "gb"}
    zsum = [sum(v) for v in zip(*zs.values())]
    c1_mean = np.array([float(a.mean()) for a in zsum])
    c1_top5 = np.array([M.topk_mean(a, 5) for a in zsum])
    a, r1 = auc_r1(yte, c1_mean[te]); a2, r12 = auc_r1(yte, c1_top5[te])
    R["m1"] = {"mean": [a, r1], "top5": [a2, r12]}
    print(f"  ① MEM mean AUC={a:.4f}  top5 AUC={a2:.4f}", flush=True)

    # ========== 方法② 统计+iForest ==========
    stats = np.array([stats_feats(u) for u in users])
    sc_s = StandardScaler().fit(stats[is_tr])
    S, Str = sc_s.transform(stats), sc_s.transform(stats[is_tr])
    ifo = IsolationForest(n_estimators=300, random_state=SEED).fit(Str)
    c2 = -ifo.score_samples(S)
    a, r1 = auc_r1(yte, c2[te])
    R["m2"] = {"auc": [a, r1]}
    print(f"  ② iForest AUC={a:.4f}", flush=True)

    # ========== 方法③ 原型距离 ==========
    emb = embed(model, users, device)
    sc_e = StandardScaler().fit(emb[is_tr])
    E, Etr = sc_e.transform(emb), sc_e.transform(emb[is_tr])
    km16 = KMeans(16, n_init=10, random_state=SEED).fit(Etr)
    c3 = np.linalg.norm(E[:, None] - km16.cluster_centers_[None], axis=2).min(1)
    km1 = KMeans(1, n_init=10, random_state=SEED).fit(Etr)
    c3_k1 = np.linalg.norm(E - km1.cluster_centers_[0], axis=1)
    a, r1 = auc_r1(yte, c3[te]); a1, _ = auc_r1(yte, c3_k1[te])
    R["m3"] = {"k16": [a, r1], "k1_auc": a1}
    print(f"  ③ 原型K=16 AUC={a:.4f} (单中心K=1对照 {a1:.4f})", flush=True)

    # ========== 方法④a 第5近白距离 ==========
    nn5 = NearestNeighbors(n_neighbors=5).fit(Str)
    c4 = nn5.kneighbors(S)[0][:, -1]
    a, r1 = auc_r1(yte, c4[te])
    R["m4a"] = {"auc": [a, r1]}
    print(f"  ④a 第5近白距离 AUC={a:.4f}", flush=True)

    # ========== 方法⑦ 秩融合 (推荐三通道: ①mean+②+④a) ==========
    def fuse(chs):
        return sum(rankdata(c[te]) for c in chs)
    f3 = fuse([c1_mean, c2, c4])
    fall = fuse([c1_mean, c2, c3, c4])
    a3, r13 = auc_r1(yte, f3); aa, r1a = auc_r1(yte, fall)
    R["m7"] = {"three": [a3, r13], "all": [aa, r1a],
               "singles": {"①MEM(mean)": R["m1"]["mean"][0],
                           "②iForest": R["m2"]["auc"][0],
                           "③原型K16": R["m3"]["k16"][0],
                           "④a近邻": R["m4a"]["auc"][0]}}
    print(f"  ⑦ 三通道融合 AUC={a3:.4f}  全通道 AUC={aa:.4f}", flush=True)

    # 报警池: 融合分在测试白样本的 p99 为阈值
    thr = np.percentile(f3[yte == 0], 99)
    alarm = f3 > thr
    n_alarm = int(alarm.sum()); n_black_hit = int((alarm & (yte == 1)).sum())
    R["alarm"] = {"thr_pct": 99, "n_alarm": n_alarm,
                  "n_black_hit": n_black_hit,
                  "black_total": int(yte.sum()),
                  "fp": n_alarm - n_black_hit}
    print(f"  报警池(白p99阈值): 报 {n_alarm} 个, 命中黑 {n_black_hit}/{int(yte.sum())}",
          flush=True)

    # ========== 方法⑤ 嵌入聚人群 ==========
    wmask = y == 0
    km4 = KMeans(4, n_init=10, random_state=SEED).fit(E[wmask])
    lab_w = km4.labels_
    ari_w = adjusted_rand_score(wtype[wmask], lab_w)
    clusters = []
    uw = [u for u in users if u["label"] == 0]
    for k in range(4):
        idx = [i for i, l in enumerate(lab_w) if l == k]
        types = [wtype[wmask][i] for i in idx]
        maj = max(set(types), key=types.count)
        night = np.mean([sum(1 for h in uw[i]["hour"] if h < 6) /
                         len(uw[i]["hour"]) for i in idx])
        ch = [c for i in idx for c in uw[i]["ch"] if c]
        topch = {1: "APP", 2: "WEB", 3: "POS"}[max(set(ch), key=ch.count)]
        clusters.append({"k": k, "n": len(idx), "majority": maj,
                         "purity": types.count(maj) / len(idx),
                         "night": float(night), "top_channel": topch})
    R["m5"] = {"ari": float(ari_w), "clusters": clusters}
    print(f"  ⑤ 人群聚类 ARI={ari_w:.4f}", flush=True)

    # ========== 方法⑥ 指纹聚手法 (报警池里的黑样本) ==========
    fp = surprise_profile(users, zs)
    te_idx = np.where(te)[0]
    alarm_black_idx = te_idx[alarm & (yte == 1)]
    Xf = StandardScaler().fit_transform(fp[alarm_black_idx])
    lab_b = KMeans(4, n_init=10, random_state=SEED).fit_predict(Xf)
    yb = btype[alarm_black_idx]
    ari_b = adjusted_rand_score(yb, lab_b)
    conf = {}
    for bt in sorted(set(yb)):
        conf[bt] = [int(sum(1 for l, b in zip(lab_b, yb)
                            if b == bt and l == k)) for k in range(4)]
    R["m6"] = {"n_alarm_black": len(alarm_black_idx),
               "ari": float(ari_b), "confusion": conf}
    print(f"  ⑥ 指纹聚手法(报警黑{len(alarm_black_idx)}个) ARI={ari_b:.4f}", flush=True)

    # ========== 方法④b 案件库检索 ==========
    rs = np.random.RandomState(SEED)
    bidx = np.where(y == 1)[0]
    known = rs.choice(bidx, 120, replace=False)
    query = np.array([i for i in bidx if i not in set(known)])
    lib = np.concatenate([np.where(is_tr)[0], known])
    lib_black = np.isin(lib, known)
    sc_f = StandardScaler().fit(fp[lib])
    nn10 = NearestNeighbors(n_neighbors=10).fit(sc_f.transform(fp[lib]))
    _, idx10 = nn10.kneighbors(sc_f.transform(fp[query]))
    hit = 0
    for qi, row in zip(query, idx10):
        nb = [btype[lib[j]] for j in row if lib_black[j]]
        if nb and max(set(nb), key=nb.count) == btype[qi]:
            hit += 1
    rec10 = float(np.mean([lib_black[row].any() for row in idx10]))
    # 干净的检测口径: 查询=测试集里不在库中的账户
    qt = np.array([i for i in te_idx if i not in set(known)])
    _, idxq = nn10.kneighbors(sc_f.transform(fp[qt]))
    s_ret = lib_black[idxq].mean(1)
    a_ret, _ = auc_r1(y[qt], s_ret)
    R["m4b"] = {"recall10": rec10, "attr_hit": hit / len(query),
                "auc": float(a_ret), "lib_size": 120}
    print(f"  ④b 案件库(120案): Recall@10={rec10:.1%} 手法命中={hit/len(query):.1%} "
          f"检测AUC={a_ret:.4f}", flush=True)

    # ========== 逐账户示例 (每手法取融合分最高的一个) ==========
    ex = []
    for bt in ("盗号爆发", "慢速抽干", "养卡套现", "赌博出款"):
        cand = [(f3[j], j) for j, i in enumerate(te_idx) if btype[i] == bt]
        _, j = max(cand)
        i = te_idx[j]
        pct = lambda c: float((rankdata(c[te])[j]) / te.sum() * 100)
        ex.append({"uid": users[i]["uid"], "btype": bt,
                   "wtype": users[i]["wtype"],
                   "pct_m1": pct(c1_mean), "pct_m2": pct(c2),
                   "pct_m4": pct(c4),
                   "alarmed": bool(f3[j] > thr)})
    # 一个正常账户对照
    jw = int(np.where(yte == 0)[0][0])
    ex.append({"uid": users[te_idx[jw]]["uid"], "btype": "(正常)",
               "wtype": users[te_idx[jw]]["wtype"],
               "pct_m1": float(rankdata(c1_mean[te])[jw] / te.sum() * 100),
               "pct_m2": float(rankdata(c2[te])[jw] / te.sum() * 100),
               "pct_m4": float(rankdata(c4[te])[jw] / te.sum() * 100),
               "alarmed": bool(f3[jw] > thr)})
    R["examples"] = ex

    json.dump(R, open("system_result.json", "w"), ensure_ascii=False, indent=1)
    print("完成 → system_result.json", flush=True)


if __name__ == "__main__":
    main()
