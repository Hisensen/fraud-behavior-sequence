# -*- coding: utf-8 -*-
"""从 7 个数据集各抽一黑一白完整序列(原始+编码), 导出 showcase JSON 供展示页用"""
import json
import random
from datetime import datetime, timezone

import numpy as np

OUT = ("/private/tmp/claude-501/-Users-macbookpro-Desktop-claude-project----/"
       "fb8a0fc5-ded6-4214-8a76-d7a5b2730fc3/scratchpad/showcase.json")
SEED = 42
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_LBL = ["≤1m", "1-5m", "5-30m", "0.5-1h", "1-6h", "6-24h", "1-7d", ">7d", "首"]


def gap_disp(sec):
    if sec is None:
        return "—"
    if sec < 60: return f"{int(sec)}s"
    if sec < 3600: return f"{sec/60:.0f}m"
    if sec < 86400: return f"{sec/3600:.1f}h"
    return f"{sec/86400:.1f}d"


def ts2str(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M")


datasets = []

# ---------- 1. temporal 合成 ----------
def do_temporal():
    from mem_experiment import TYPE2ID, SENSITIVE, GAP_BOS, amount_bucket, gap_bucket
    samples = []
    for line in open("data_temporal.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["user_id"] not in ("B_0291", "W_0485"):
            continue
        evs, prev = [], None
        for ev in r["events"]:
            t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
            gap = (t - prev).total_seconds() if prev else None
            gi = GAP_BOS if gap is None else gap_bucket(gap)
            evs.append({"t": ev["t"][5:16], "name": ev["type"],
                        "amt": ev.get("amount"), "gap": gap_disp(gap),
                        "enc": [TYPE2ID[ev["type"]], amount_bucket(ev), int(gi)],
                        "fraud": 1 if (r["label"] == 1 and ev["type"] in SENSITIVE) else 0})
            prev = t
        samples.append({"label": r["label"], "uid": r["user_id"], "events": evs})
    samples.sort(key=lambda s: -s["label"])
    datasets.append({
        "id": "temporal", "title": "temporal 合成数据", "real": "合成",
        "task": "用户级 · 盗号/套现/洗钱 vs 正常",
        "event_def": "一个事件 = 一次 App 操作(登录/消费/改密码等 12 类)",
        "enc_def": "编码 = [事件类型 0-11, 金额档 0-3, 间隔桶 0-8]；黑样本高亮 = 敏感操作",
        "samples": samples})


# ---------- 2. Ethereum ----------
def do_eth():
    from eth_experiment import build_encoder, GAP_BOS as GB
    users = [json.loads(l) for l in open("eth_sequences.jsonl")]
    users = [u for u in users if len(u["txs"]) >= 10]
    normal = [u for u in users if u["label"] == 0]
    rng = random.Random(SEED); rng.shuffle(normal)
    enc = build_encoder(normal[:4000])
    def pick(lst):
        return next(u for u in lst if 40 <= len(u["txs"]) <= 100)
    samples = []
    for u in (pick([x for x in users if x["label"] == 1]), pick(normal[4000:])):
        e = enc(u)
        evs, prev = [], None
        for k, (ts, d, a) in enumerate(u["txs"][-120:]):
            gap = ts - prev if prev else None
            evs.append({"t": ts2str(ts), "name": "转出" if d else "转入",
                        "amt": round(a, 4), "gap": gap_disp(gap),
                        "enc": [e["types"][k], e["amounts"][k], e["gaps"][k]],
                        "fraud": 0})
            prev = ts
        samples.append({"label": u["label"], "uid": u["addr"][:14] + "…",
                        "events": evs})
    datasets.append({
        "id": "eth", "title": "Ethereum 钓鱼(XBlock)", "real": "✅ 真实",
        "task": "账户级 · 钓鱼收款账户 vs 正常账户(账户整体标注, 无逐笔标)",
        "event_def": "一个事件 = 一笔链上转账(方向 + ETH 金额 + 时间)",
        "enc_def": "编码 = [方向 0转入/1转出, 金额档 0-7, 间隔桶 0-8]",
        "samples": samples})


# ---------- 3/4. Sparkov & BankSim (窗口型) ----------
def do_sparkov():
    from sparkov_experiment import load_windows, encode_all
    wins = load_windows()
    tr, tw, bl = encode_all(wins)
    raw_b = wins["black"]; raw_w = wins["test_w"][:1500]
    bi = next(i for i, w in enumerate(raw_b) if 3 <= w["fr"].sum() <= 8)
    wi = 0
    samples = []
    for raw, e, lab in ((raw_b[bi], bl[bi], 1), (raw_w[wi], tw[wi], 0)):
        evs, prev = [], None
        for k in range(len(raw["ts"])):
            ts = int(raw["ts"][k]); gap = ts - prev if prev else None
            evs.append({"t": ts2str(ts), "name": str(raw["cat"][k]).replace("_", " "),
                        "amt": round(float(raw["amt"][k]), 2), "gap": gap_disp(gap),
                        "enc": [e["types"][k], e["amounts"][k], e["gaps"][k]],
                        "fraud": int(raw["fr"][k])})
            prev = ts
        samples.append({"label": lab, "uid": f"窗口(客户{raw.get('user','')})",
                        "events": evs})
    datasets.append({
        "id": "sparkov", "title": "Sparkov 信用卡", "real": "拟真",
        "task": "窗口级(64笔) · 含欺诈窗口 vs 干净窗口, 逐笔标注",
        "event_def": "一个事件 = 一笔刷卡消费(商户类别 + 金额 + 时间)",
        "enc_def": "编码 = [类别 0-14, 金额档 0-7, 间隔桶 0-8]；红行 = is_fraud=1",
        "samples": samples})


def do_banksim():
    from banksim_experiment import load_windows, encode_all
    wins = load_windows()
    tr, tw, bl = encode_all(wins)
    raw_b = wins["black"]; raw_w = wins["test_w"][:1500]
    bi = next(i for i, w in enumerate(raw_b) if 3 <= w["fr"].sum() <= 8)
    samples = []
    for raw, e, lab in ((raw_b[bi], bl[bi], 1), (raw_w[0], tw[0], 0)):
        evs, prev = [], None
        for k in range(len(raw["st"])):
            st = int(raw["st"][k]); gap = (st - prev) if prev is not None else None
            evs.append({"t": f"第{st}天", "name": str(raw["ca"][k]).replace("es_", ""),
                        "amt": round(float(raw["am"][k]), 2),
                        "gap": "—" if gap is None else f"{gap}天",
                        "enc": [e["types"][k], e["amounts"][k], e["gaps"][k]],
                        "fraud": int(raw["fr"][k])})
            prev = st
        samples.append({"label": lab, "uid": "窗口", "events": evs})
    datasets.append({
        "id": "banksim", "title": "BankSim 银行支付", "real": "拟真",
        "task": "窗口级(64笔) · 逐笔标注; 时间仅天粒度",
        "event_def": "一个事件 = 一笔支付(商户类别 + 金额 + 天序号)",
        "enc_def": "编码 = [类别 0-15, 金额档 0-7, 间隔桶(天) 0-6]；红行 = fraud=1",
        "samples": samples})


# ---------- 5. PaySim (收款账户) ----------
def do_paysim():
    import pandas as pd
    CSV = ("/Users/macbookpro/.cache/kagglehub/datasets/ealaxi/paysim1/"
           "versions/2/PS_20174392719_1491204439457_log.csv")
    df = pd.read_csv(CSV, usecols=["step", "nameOrig", "nameDest", "type",
                                   "amount", "isFraud"])
    vcd = df["nameDest"].value_counts()
    fr_counts = df[df.isFraud == 1].groupby("nameDest").size()
    cand = sorted(fr_counts.index,
                  key=lambda a: (-min(fr_counts[a], 3), -vcd.get(a, 0)))
    black_acct = next((a for a in cand if 10 <= vcd.get(a, 0) <= 120), cand[0])
    fraud_dest = set(df.loc[df.isFraud == 1, "nameDest"])
    white_acct = next(a for a in vcd.index
                      if 20 <= vcd[a] <= 60 and a not in fraud_dest)
    from paysim_experiment import GAP_EDGES_H, GAP_BOS as GB, N_AMOUNT
    amts = df["amount"].to_numpy(); amts = amts[amts > 0]
    q = np.quantile(np.random.default_rng(0).choice(amts, 200000),
                    np.linspace(0, 1, N_AMOUNT)[1:-1])
    samples = []
    for acct, lab in ((black_acct, 1), (white_acct, 0)):
        a = df[df.nameDest == acct]
        b = df[df.nameOrig == acct]
        rows = []
        for _, r in a.iterrows():
            rows.append((r.step, r.type, 0, r.amount, r.isFraud))
        for _, r in b.iterrows():
            rows.append((r.step, r.type, 1, r.amount, 0))
        rows.sort()
        evs, prev = [], None
        for step, ty, d, amt, isf in rows[:120]:
            gap = step - prev if prev is not None else None
            gi = GB if gap is None else int(np.searchsorted(GAP_EDGES_H, max(gap, 0),
                                                            side="left"))
            ai = 0 if amt <= 0 else 1 + int(np.searchsorted(q, amt))
            evs.append({"t": f"第{step}小时", "name": f"{ty}{'·发起' if d else '·收到'}",
                        "amt": round(float(amt), 2),
                        "gap": "—" if gap is None else f"{gap}h",
                        "enc": [f"{ty[:4]}×{d}", ai, gi], "fraud": int(isf)})
            prev = step
        samples.append({"label": lab, "uid": acct, "events": evs})
    datasets.append({
        "id": "paysim", "title": "PaySim 手机转账(收款账户视角)", "real": "拟真",
        "task": "账户级 · 欺诈收款账户 vs 正常; 时间仅小时粒度 ❌实测无信号",
        "event_def": "一个事件 = 该账户参与的一笔转账(类型×方向 + 金额)",
        "enc_def": "编码 = [类型×方向, 金额档 0-7, 间隔桶(小时) 0-8]；红行 = isFraud=1",
        "samples": samples})


# ---------- 6. IBM AML ----------
def do_aml():
    import pandas as pd
    from aml_experiment import find_csv, GAP_BOS as GB, N_AMOUNT
    df = pd.read_csv(find_csv(), usecols=["Timestamp", "Account", "Account.1",
                                          "Amount Paid", "Payment Format",
                                          "Is Laundering"])
    df.columns = ["ts", "src", "dst", "amt", "fmt", "y"]
    df["ts"] = pd.to_datetime(df["ts"]).astype("int64") // 10**9
    dirty_cnt = pd.concat([df.loc[df.y == 1, "src"],
                           df.loc[df.y == 1, "dst"]]).value_counts()
    cnt = pd.concat([df["src"], df["dst"]]).value_counts()
    black_acct = next(a for a in dirty_cnt.index
                      if 25 <= cnt.get(a, 0) <= 80 and dirty_cnt[a] >= 3)
    dirty = set(df.loc[df.y == 1, "src"]) | set(df.loc[df.y == 1, "dst"])
    white_acct = next(a for a in cnt.index
                      if 25 <= cnt[a] <= 80 and a not in dirty)
    amts = df["amt"].to_numpy(); amts = amts[amts > 0]
    q = np.quantile(np.random.default_rng(0).choice(amts, 200000),
                    np.linspace(0, 1, N_AMOUNT)[1:-1])
    samples = []
    for acct, lab in ((black_acct, 1), (white_acct, 0)):
        a = df[df.dst == acct]; b = df[df.src == acct]
        rows = ([(r.ts, r.fmt, 0, r.amt, r.y) for _, r in a.iterrows()] +
                [(r.ts, r.fmt, 1, r.amt, r.y) for _, r in b.iterrows()])
        rows.sort()
        evs, prev = [], None
        for ts, fmt, d, amt, isf in rows[:120]:
            gap = ts - prev if prev else None
            gi = GB if gap is None else int(np.searchsorted(GAP_EDGES, max(gap, 0),
                                                            side="left"))
            ai = 0 if amt <= 0 else 1 + int(np.searchsorted(q, amt))
            evs.append({"t": ts2str(ts), "name": f"{fmt}{'·付出' if d else '·收到'}",
                        "amt": round(float(amt), 2), "gap": gap_disp(gap),
                        "enc": [f"{fmt[:5]}×{d}", ai, gi], "fraud": int(isf)})
            prev = ts
        samples.append({"label": lab, "uid": str(acct), "events": evs})
    datasets.append({
        "id": "aml", "title": "IBM AML 转账流水", "real": "拟真",
        "task": "账户级 · 洗钱账户 vs 正常; 红行 = Is Laundering=1 ⚠️信号主要在图",
        "event_def": "一个事件 = 一笔银行转账(支付方式×方向 + 金额)",
        "enc_def": "编码 = [方式×方向, 金额档 0-7, 间隔桶 0-8]",
        "samples": samples})


for fn in (do_temporal, do_eth, do_sparkov, do_banksim, do_paysim, do_aml):
    print("running", fn.__name__, flush=True)
    fn()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"datasets": datasets, "gap_labels": GAP_LBL}, f, ensure_ascii=False)
n = sum(len(s["events"]) for d in datasets for s in d["samples"])
print(f"完成: {len(datasets)} 数据集, 共 {n} 行事件 → {OUT}")
