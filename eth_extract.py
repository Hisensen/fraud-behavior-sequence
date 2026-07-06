# -*- coding: utf-8 -*-
"""
从 XBlock EPTransNet 大图(297万节点/1355万边)抽取账户级交易序列。
输出 eth_sequences.jsonl: {"addr", "label", "txs": [[ts, dir, amount], ...]}
  dir: 0=转入(别人→我), 1=转出(我→别人)
账户筛选: 交易数 ∈ [10, 2000], 序列截取最近 200 笔。
正常账户从 isp=0 节点随机采样 5000 个(注意: 该网络由钓鱼种子 BFS 爬出,
"正常"节点实为钓鱼账户的 1-2 阶邻居, 采样偏置在报告中说明)。
"""
import json
import pickle
import random
import sys

random.seed(42)

PKL = ("/Users/macbookpro/.cache/kagglehub/datasets/xblock/"
       "ethereum-phishing-transaction-network/versions/1/"
       "Ethereum Phishing Transaction Network/MulDiGraph.pkl")

print("加载大图(峰值内存约10GB, 请等待)...", flush=True)
with open(PKL, "rb") as f:
    G = pickle.load(f)
print("节点", G.number_of_nodes(), "边", G.number_of_edges(), flush=True)

# 探测边属性字段名
u, v, attr = next(iter(G.edges(data=True)))
print("边属性样例:", attr, flush=True)
AMT_KEY = "amount" if "amount" in attr else ("value" if "value" in attr else list(attr)[0])
TS_KEY = "timestamp" if "timestamp" in attr else ("time" if "time" in attr else list(attr)[-1])

phish = [n for n, d in G.nodes(data=True) if d.get("isp", 0) == 1]
print("钓鱼节点:", len(phish), flush=True)

def txs_of(node):
    txs = []
    for _, _, d in G.out_edges(node, data=True):
        txs.append((float(d[TS_KEY]), 1, float(d[AMT_KEY])))
    for _, _, d in G.in_edges(node, data=True):
        txs.append((float(d[TS_KEY]), 0, float(d[AMT_KEY])))
    txs.sort()
    return txs

out = open("eth_sequences.jsonl", "w")
kept_p = 0
for n in phish:
    deg = G.out_degree(n) + G.in_degree(n)
    if not (10 <= deg <= 2000):
        continue
    txs = txs_of(n)[-200:]
    out.write(json.dumps({"addr": n, "label": 1, "txs": txs}) + "\n")
    kept_p += 1
print("保留钓鱼账户:", kept_p, flush=True)

# 正常账户采样: 先按度粗筛再精筛, 避免全图遍历两次
normals = []
nodes = list(G.nodes())
random.shuffle(nodes)
for n in nodes:
    if len(normals) >= 5000:
        break
    if G.nodes[n].get("isp", 0) == 1:
        continue
    deg = G.out_degree(n) + G.in_degree(n)
    if 10 <= deg <= 2000:
        normals.append(n)
for n in normals:
    txs = txs_of(n)[-200:]
    out.write(json.dumps({"addr": n, "label": 0, "txs": txs}) + "\n")
out.close()
print("保留正常账户:", len(normals), flush=True)
print("完成 → eth_sequences.jsonl", flush=True)
