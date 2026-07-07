# -*- coding: utf-8 -*-
"""数据流全程追踪: 原始→编码→模型→惊讶度→z分→池化→判定→定位, 逐站打印真实数值"""
import json
import random

import numpy as np
import torch

from fraud_pipeline import (load_jsonl, Encoder, build_model, per_position_ce,
                            apply_norm, GAP_LBL, SEED)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def trace(data_path, model_path, uid_black, uid_white, shape, n_show=8):
    ck = torch.load(model_path)
    enc = Encoder.from_meta(ck["meta"])
    model = build_model(shape, enc)
    model.load_state_dict(ck["state"])
    model.eval()
    ns = ck["norm_stats"]
    thr = ck["thr"]
    cn, an = ck["best"].split("|")
    users = {u["uid"]: u for u in load_jsonl(data_path)
             if u["uid"] in (uid_black, uid_white)}

    for uid, tag in ((uid_black, "黑样本"), (uid_white, "白样本")):
        u = users[uid]
        print("=" * 66)
        print(f"【{tag}】 {uid[:28]}…  ({len(u['events'])} 个事件)")
        print("=" * 66)

        # ---- 站点1: 原始 ----
        print(f"\n── 站点1 · 原始数据(前{n_show}个事件) " + "─" * 20)
        for i, ev in enumerate(u["events"][:n_show]):
            amt = f"  amount={ev['amount']}" if "amount" in ev else ""
            print(f"  #{i:<3} {ev['t']}  {ev['type']}{amt}")

        # ---- 站点2: 编码 ----
        e = enc.encode(u)
        print(f"\n── 站点2 · 编码后(同样前{n_show}个位置的整数数组) " + "─" * 6)
        print(f"  {'#':<4}{'type':>5}{'gamt':>6}{'pamt':>6}{'gap':>5}{'hour':>7}")
        for i in range(min(n_show, len(e["type"]))):
            print(f"  {i:<4}{e['type'][i]:>5}{e['gamt'][i]:>6}"
                  f"{e['pamt'][i]:>6}{e['gap'][i]:>5}{e['hour'][i]:>7.1f}")
        print(f"  (type: 0={list(enc.type_vocab)[0]} 1={list(enc.type_vocab)[1] if len(enc.type_vocab)>1 else '—'}"
              f" | pamt: 金额÷个人中位数的倍数档 | gap桶: {'/'.join(GAP_LBL[:4])}...)")

        # ---- 站点3: 模型逐位置惊讶度 ----
        pcs = per_position_ce(model, [e], shape, "cpu")
        heads = model.heads_list
        print(f"\n── 站点3 · 模型输出: 逐位置惊讶度(交叉熵), 每头一行 " + "─" * 2)
        L = len(e["type"])
        show_idx = list(range(min(n_show, L)))
        for h in heads:
            vals = " ".join(f"{pcs[0][h][i]:5.2f}" for i in show_idx)
            print(f"  {h:<5}: {vals} ...")

        # ---- 站点4: z-norm ----
        print(f"\n── 站点4 · 难度归一化后的 z 分(同样位置) " + "─" * 12)
        zs = {h: apply_norm(pcs, h, np.array(ns[h][0]), np.array(ns[h][1]))
              for h in heads}
        for h in heads:
            vals = " ".join(f"{zs[h][0][i]:5.2f}" for i in show_idx)
            print(f"  z_{h:<4}: {vals} ...")

        # ---- 站点5: 池化 → 一个分数 ----
        if cn == "SUM":
            z_final = sum(zs[h][0] for h in heads)
        else:
            z_final = zs[cn][0]
        if an == "mean":
            score = float(z_final.mean())
            how = f"全序列平均({L}个位置)"
        else:
            k = int(an.replace("top", ""))
            topv = np.sort(z_final)[-k:]
            score = float(topv.mean())
            how = f"top-{k}平均: ({' + '.join(f'{v:.2f}' for v in topv)})/{k}"
        print(f"\n── 站点5 · 池化(最优口径 {ck['best']}) " + "─" * 14)
        print(f"  {how} = 风险分 {score:.3f}")

        # ---- 站点6: 判定 ----
        hit = score >= thr
        print(f"\n── 站点6 · 阈值判定 " + "─" * 26)
        print(f"  {score:.3f} {'≥' if hit else '<'} 阈值{thr:.3f}(白样本p99) "
              f"→ {'命中⚑ 进审核队列' if hit else '放行'}")

        # ---- 站点7: 定位回溯 ----
        top3 = np.argsort(-z_final)[:3]
        print(f"\n── 站点7 · 定位: z分最高的3个位置 → 回查原始事件 " + "─" * 2)
        for j in top3:
            ev = u["events"][j]
            amt = f" amount={ev.get('amount','—')}"
            print(f"  #{j:<4} z={z_final[j]:5.2f}  {ev['t']} {ev['type']}{amt}")
        print()


import csv

print("\n" + "█" * 66)
print("█  数据集一: 以太坊真实链上数据 (资金流, mean池化)")
print("█" * 66 + "\n")
rows_e = list(csv.DictReader(open("eth_outputs/scores.csv", encoding="utf-8")))
blk_e = max((r for r in rows_e if r["label"] == "1"),
            key=lambda r: float(r["score"]))
wht_e = min((r for r in rows_e if r["label"] == "0"),
            key=lambda r: float(r["score"]))
trace("eth_real.jsonl", "eth_outputs/model.pt",
      blk_e["user_id"], wht_e["user_id"], "funds")

print("\n" + "█" * 66)
print("█  数据集二: rich 操作流(盗号场景, 自回归+top-k池化)")
print("█" * 66 + "\n")
# 找一个高分黑和一个低分白
import csv
rows = list(csv.DictReader(open("demo_outputs/scores.csv", encoding="utf-8")))
blk = max((r for r in rows if r["label"] == "1"), key=lambda r: float(r["score"]))
wht = min((r for r in rows if r["label"] == "0"), key=lambda r: float(r["score"]))
trace("demo_data.jsonl", "demo_outputs/model.pt", blk["user_id"],
      wht["user_id"], "operation")
