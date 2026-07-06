# -*- coding: utf-8 -*-
"""
以太坊二次抽取: 在 v1 基础上补"对手方新旧"信息。
每笔交易输出 [ts, dir, amount, prior] —— prior = 该对手方在此账户
此前历史中已出现的次数(抽取时算好, 不存地址, 文件保持轻量)。
输出 eth_sequences2.jsonl, 账户选择逻辑与 v1 完全一致(同种子)。
"""
import json
import pickle
import random
from collections import defaultdict

random.seed(42)

PKL = ("/Users/macbookpro/.cache/kagglehub/datasets/xblock/"
       "ethereum-phishing-transaction-network/versions/1/"
       "Ethereum Phishing Transaction Network/MulDiGraph.pkl")

print("加载大图...", flush=True)
with open(PKL, "rb") as f:
    G = pickle.load(f)
print("节点", G.number_of_nodes(), "边", G.number_of_edges(), flush=True)

phish = [n for n, d in G.nodes(data=True) if d.get("isp", 0) == 1]

def txs_of(node):
    txs = []
    for _, v, d in G.out_edges(node, data=True):
        txs.append((float(d["timestamp"]), 1, float(d["amount"]), v))
    for u, _, d in G.in_edges(node, data=True):
        txs.append((float(d["timestamp"]), 0, float(d["amount"]), u))
    txs.sort(key=lambda x: x[0])
    seen = defaultdict(int)
    out = []
    for ts, dr, amt, cp in txs:
        out.append([ts, dr, amt, seen[cp]])
        seen[cp] += 1
    return out

out = open("eth_sequences2.jsonl", "w")
kept = 0
for n in phish:
    deg = G.out_degree(n) + G.in_degree(n)
    if not (10 <= deg <= 2000):
        continue
    out.write(json.dumps({"addr": n, "label": 1,
                          "txs": txs_of(n)[-200:]}) + "\n")
    kept += 1
print("钓鱼账户:", kept, flush=True)

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
    out.write(json.dumps({"addr": n, "label": 0,
                          "txs": txs_of(n)[-200:]}) + "\n")
out.close()
print("正常账户:", len(normals), "→ eth_sequences2.jsonl", flush=True)
