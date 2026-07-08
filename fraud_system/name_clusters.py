# -*- coding: utf-8 -*-
"""
流水线第 2 步: LLM 自动定类 —— 对已训练模型的聚类结果:
  人群簇(未报警账户, 嵌入空间) + 手法簇(报警池账户, 指纹空间, 簇数自动选)
  → 每簇提取无标签画像卡 → claude 盲命名 → 结果写回 models/<名>.pkl
之后 webapp 测试任何新账户, 直接输出 LLM 定好的类别。

CLI: python name_clusters.py [模型名=cluster_v1]
     (数据带真值标签时, 结尾自动打印对账表用于验证)
"""
import json
import pickle
import subprocess
import sys
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import mem_rich as M
from cluster_experiment import embed, surprise_profile

STAT_NAMES = ([f"事件占比·{t}" for t in M.EVENT_TYPES] +
              [f"间隔占比·{g}" for g in ["<1分", "1-5分", "5-30分", "0.5-1时",
                                        "1-6时", "6-24时", "1-7天", ">7天",
                                        "首笔", "其他"]] +
              [f"金额倍数占比·{p}" for p in ["无金额", "<0.25×", "0.25-0.5×",
                                            "0.5-1×", "1-2×", "2-4×", "4-8×",
                                            ">8×", "其他"]] +
              ["渠道占比·APP", "渠道占比·WEB", "渠道占比·POS",
               "凌晨0-6点事件占比", "时刻sin", "时刻cos",
               "登录失败率", "换IP率", "log(月事件数)"])
FP_HEADS = {"h_gap": "间隔头", "h_pamt": "个人金额头",
            "h_res": "结果头", "h_type": "类型头"}


def digest(rec):
    evs = rec["events"]
    tc = Counter(e["type"] for e in evs)
    hrs = Counter()
    for e in evs:
        h = int(e["t"][11:13])
        hrs["凌晨0-6" if h < 6 else "上午6-12" if h < 12
            else "下午12-18" if h < 18 else "晚间18-24"] += 1
    ch = Counter(e.get("channel", "-") for e in evs)
    amts = [e["amount"] for e in evs if "amount" in e]
    return (f"{len(evs)}笔/月 · 事件构成 " +
            " ".join(f"{t}×{c}" for t, c in tc.most_common(5)) +
            f" · 渠道 {dict(ch.most_common(3))} · 时段 " +
            " ".join(f"{k}:{v/len(evs):.0%}" for k, v in hrs.most_common()) +
            (f" · 单笔金额中位 {np.median(amts):.0f}元" if amts else ""))


def diff_lines(Xc, Xg, names, top=8):
    mu_g, sd_g = Xg.mean(0), Xg.std(0) + 1e-9
    z = (Xc.mean(0) - mu_g) / sd_g
    idx = np.argsort(-np.abs(z))[:top]
    return [f"{names[i]}: 本簇 {Xc.mean(0)[i]:.3f} vs 全体 {mu_g[i]:.3f}"
            f"（{'高' if z[i] > 0 else '低'} {abs(z[i]):.1f} 个标准差）"
            for i in idx if abs(z[i]) > 0.3]


def ask_llm(cards, log):
    prompt = ("你是银行客群与反欺诈分析师。下面是若干聚类簇的画像卡(JSON)。"
              "P 开头的是正常客户人群簇, F 开头的是报警账户的异常手法簇。"
              "每张卡只包含行为统计与代表账户的行为摘要, 没有任何标签。"
              "请只依据卡内信息, 为每个簇给出定论。严格输出 JSON 数组, "
              "每个元素: {\"id\":..., \"name\":\"簇名(≤6个汉字)\", "
              "\"definition\":\"一句话定义\", \"evidence\":[\"判别特征×3\"], "
              "\"caveat\":\"存疑点\"}。不得臆测统计之外的信息, 不要输出其他文字。\n\n"
              + json.dumps(cards, ensure_ascii=False))
    log(f"调用 claude 盲命名({len(cards)} 张画像卡)…")
    out = subprocess.run(["claude", "-p", prompt], capture_output=True,
                         text=True, timeout=600).stdout.strip()
    if out.startswith("```"):
        out = out.strip("`").lstrip("json").strip()
    return {x["id"]: x for x in json.loads(out)}


def name_bundle(model_name, log=print):
    from webapp import (MODELS_DIR, apply_znorm, encode_rec, load_bundle,
                        pct_among, stats_feats)
    import os
    b = load_bundle(model_name)
    raw = [json.loads(l) for l in open(b["data"], encoding="utf-8")]
    users = [encode_rec(r, b["gq"]) for r in raw]
    byid = {r["user_id"]: r for r in raw}

    log(f"给全部 {len(users)} 账户打分以划定报警池(最耗时的一步)…")
    pcs = M.per_position_ce(b["model"], users, "cpu")
    zs = apply_znorm(pcs, b["zn"])
    zsum = [sum(zs[k][i] for k in zs) for i in range(len(users))]
    c1m = np.array([float(a.mean()) for a in zsum])
    S = b["st_sc"].transform(np.array([stats_feats(u) for u in users]))
    c2 = -b["ifo"].score_samples(S)
    nn = NearestNeighbors(n_neighbors=5).fit(b["nn_ref"])
    c4 = nn.kneighbors(S)[0][:, -1]
    fused = np.array([np.mean([pct_among(c1m[i], b["c1m_w"]),
                               pct_among(c2[i], b["c2_w"]),
                               pct_among(c4[i], b["c4_w"])])
                      for i in range(len(users))])
    alarm = fused > b["thr"]
    log(f"报警池 {int(alarm.sum())} 个 / 未报警 {int((~alarm).sum())} 个")

    cards, meta = [], {}
    # ---------- 人群簇: 未报警账户, 已训练的簇心 ----------
    ok_i = np.where(~alarm)[0]
    E = b["emb_sc"].transform(embed(b["model"], [users[i] for i in ok_i], "cpu"))
    lab = np.linalg.norm(E[:, None] - b["km_centers"][None], axis=2).argmin(1)
    Sw = np.array([stats_feats(users[i]) for i in ok_i])
    for k in range(len(b["km_centers"])):
        m = lab == k
        med = np.where(m)[0][np.argsort(
            np.linalg.norm(E[m] - b["km_centers"][k], axis=1))[:2]]
        cards.append({"id": f"P{k}", "kind": "正常客户人群簇",
                      "n": int(m.sum()),
                      "差异统计": diff_lines(Sw[m], Sw, STAT_NAMES),
                      "代表账户": [digest(byid[users[ok_i[j]]["uid"]])
                                   for j in med]})
        meta[f"P{k}"] = [users[ok_i[j]]["uid"] for j in np.where(m)[0]]

    # ---------- 手法簇: 报警池账户, 指纹空间, 簇数轮廓系数自选 ----------
    al_i = np.where(alarm)[0]
    al_users = [users[i] for i in al_i]
    fp = surprise_profile(al_users, {k: [zs[k][i] for i in al_i] for k in zs})
    fp_sc = StandardScaler().fit(fp)
    X = fp_sc.transform(fp)
    best = (None, -2)
    for kk in range(2, 7):
        lb = KMeans(kk, n_init=10, random_state=42).fit_predict(X)
        s = silhouette_score(X, lb)
        if s > best[1]:
            best = (kk, s)
    n_fp = best[0]
    log(f"手法簇数(轮廓系数自选): {n_fp}")
    km = KMeans(n_fp, n_init=10, random_state=42).fit(X)
    heads = sorted(zs)
    fp_names = ([f"{FP_HEADS[h]}·{s}" for h in heads
                 for s in ("平均z", "top5均z", "最大z", "超2σ占比")] +
                [f"高惊讶事件占比·{t}" for t in M.EVENT_TYPES] +
                [f"高惊讶节奏·{g}" for g in ["<1分", "1-5分", "5-30分",
                                            "0.5-1时", "1-6时", "6-24时",
                                            "1-7天", ">7天", "首笔", "其他"]])
    for k in range(n_fp):
        m = km.labels_ == k
        med = np.where(m)[0][np.argsort(
            np.linalg.norm(X[m] - km.cluster_centers_[k], axis=1))[:2]]
        stories = []
        for j in med:
            zj = zsum[al_i[j]]
            top = sorted(np.argsort(-zj)[:5].tolist())
            evs = byid[al_users[j]["uid"]]["events"][:M.MAX_LEN]
            stories.append("; ".join(
                f"[{evs[i]['t'][5:16]}] {evs[i]['type']}"
                + (f" {evs[i]['amount']:.0f}元" if 'amount' in evs[i] else "")
                + f"(z={zj[i]:.1f})" for i in top))
        cards.append({"id": f"F{k}", "kind": "报警账户异常手法簇",
                      "n": int(m.sum()),
                      "异常指纹差异": diff_lines(fp[m], fp, fp_names),
                      "代表账户整体画像": [digest(byid[al_users[j]["uid"]])
                                          for j in med],
                      "代表账户最可疑5笔": stories})
        meta[f"F{k}"] = [al_users[j]["uid"] for j in np.where(m)[0]]

    open("cluster_cards.md", "w", encoding="utf-8").write(
        json.dumps(cards, ensure_ascii=False, indent=1))
    names = ask_llm(cards, log)
    json.dump(names, open("cluster_names.json", "w"), ensure_ascii=False,
              indent=1)

    # ---------- 写回模型 ----------
    disk = pickle.load(open(os.path.join(MODELS_DIR, model_name + ".pkl"), "rb"))
    disk["pop_names"] = {int(k[1:]): v for k, v in names.items()
                         if k.startswith("P")}
    disk["fp_km"] = {"scaler": fp_sc, "centers": km.cluster_centers_,
                     "names": {int(k[1:]): v for k, v in names.items()
                               if k.startswith("F")}}
    pickle.dump(disk, open(os.path.join(MODELS_DIR, model_name + ".pkl"), "wb"))
    log("✅ 类别已写回模型, 之后测试新账户将直接输出这些 LLM 类别")
    for cid, x in sorted(names.items()):
        log(f"  {cid} {x['name']} —— {x['definition']}")

    # ---------- 有真值时对账(仅验证用) ----------
    if any(r.get("wtype") for r in raw):
        log("---- 对账(数据带真值, 仅验证): ----")
        for cid, uids in meta.items():
            key = "wtype" if cid.startswith("P") else "btype"
            ts = [byid[u].get(key) for u in uids if byid[u].get(key)]
            maj = max(set(ts), key=ts.count) if ts else "?"
            log(f"  {cid} LLM={names[cid]['name']:<8} 真值多数={maj} "
                f"(纯度{ts.count(maj)/len(ts):.0%})" if ts else f"  {cid} 无真值")
    return names


if __name__ == "__main__":
    name_bundle(sys.argv[1] if len(sys.argv) > 1 else "cluster_v1")
