# -*- coding: utf-8 -*-
"""导出交互式追踪器所需的完整轨迹 JSON: 5 个真实账户 × 七站全数据"""
import csv
import json
import random

import numpy as np
import torch

from fraud_pipeline import (load_jsonl, Encoder, build_model, per_position_ce,
                            apply_norm, SEED)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = ("/private/tmp/claude-501/-Users-macbookpro-Desktop-claude-project----/"
       "fb8a0fc5-ded6-4214-8a76-d7a5b2730fc3/scratchpad/traces.json")


def build(data_path, model_path, shape, uid_list, ds_name):
    ck = torch.load(model_path)
    enc = Encoder.from_meta(ck["meta"])
    model = build_model(shape, enc)
    model.load_state_dict(ck["state"])
    model.eval()
    ns = ck["norm_stats"]
    users = {u["uid"]: u for u in load_jsonl(data_path)}
    out = []
    id2type = {v: k for k, v in enc.type_vocab.items()}
    for uid, note in uid_list:
        u = users[uid]
        e = enc.encode(u)
        pcs = per_position_ce(model, [e], shape, "cpu")
        heads = model.heads_list
        z = {h: apply_norm(pcs, h, np.array(ns[h][0]),
                           np.array(ns[h][1]))[0] for h in heads}
        cn, an = ck["best"].split("|")
        zf = sum(z[h] for h in heads) if cn == "SUM" else z[cn]
        if an == "mean":
            score = float(zf.mean())
            pool = {"how": "mean", "k": len(zf)}
        else:
            k = int(an.replace("top", ""))
            idx = np.argsort(-zf)[:k]
            score = float(zf[idx].mean())
            pool = {"how": "topk", "k": k,
                    "vals": [round(float(zf[i]), 2) for i in np.sort(idx)]}
        top = [int(i) for i in np.argsort(-zf)[:3]]
        events = []
        for ev in u["events"]:
            events.append({"t": ev["t"][5:16], "ty": ev["type"],
                           "a": ev.get("amount")})
        out.append({
            "id": uid, "ds": ds_name, "label": u["label"], "note": note,
            "score": round(score, 3), "thr": round(float(ck["thr"]), 3),
            "combo": ck["best"], "heads": heads, "pool": pool,
            "events": events,
            "enc": {k: [int(x) for x in e[k]] for k in
                    ("type", "gamt", "pamt", "gap", "res", "ch", "ip")},
            "hour": [round(float(h), 1) for h in e["hour"]],
            "ce": {h: [round(float(x), 2) for x in pcs[0][h]] for h in heads},
            "z": {h: [round(float(x), 2) for x in z[h]] for h in heads},
            "zf": [round(float(x), 2) for x in zf],
            "top": top,
            "typemap": {str(v): k for k, v in enc.type_vocab.items()},
        })
        print(f"  {ds_name} {uid[:20]}… score={score:.3f} thr={ck['thr']:.3f}")
    return out


traces = []
# 以太坊: 高分钓鱼 / 被误伤的正常 / 安全正常
rows = list(csv.DictReader(open("eth_outputs/scores.csv", encoding="utf-8")))
blk = max((r for r in rows if r["label"] == "1"), key=lambda r: float(r["score"]))
fpw = max((r for r in rows if r["label"] == "0"), key=lambda r: float(r["score"]))
low = min((r for r in rows if r["label"] == "0"), key=lambda r: float(r["score"]))
traces += build("eth_real.jsonl", "eth_outputs/model.pt", "funds",
                [(blk["user_id"], "真实钓鱼地址·全场最高分"),
                 (fpw["user_id"], "正常账户·被误伤(那1%的代价)"),
                 (low["user_id"], "正常账户·全场最低分")], "以太坊·真实资金流")

rows = list(csv.DictReader(open("demo_outputs/scores.csv", encoding="utf-8")))
blk = max((r for r in rows if r["label"] == "1"), key=lambda r: float(r["score"]))
wht = min((r for r in rows if r["label"] == "0"), key=lambda r: float(r["score"]))
traces += build("demo_data.jsonl", "demo_outputs/model.pt", "operation",
                [(blk["user_id"], "盗号账户·含攻击链"),
                 (wht["user_id"], "正常账户")], "rich·操作流(盗号)")

json.dump(traces, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
import os
print(f"\n{len(traces)} 条轨迹 → {OUT} ({os.path.getsize(OUT)//1024}KB)")
