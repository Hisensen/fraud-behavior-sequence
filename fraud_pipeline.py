# -*- coding: utf-8 -*-
"""
行为序列反欺诈 · 统一流水线工具
================================
面向真实业务数据(格式未知)的四步操作:

  python fraud_pipeline.py inspect  原始数据.csv              # ① 看字段, 生成映射模板
  python fraud_pipeline.py convert  原始数据.csv -m mapping.json -o data.jsonl   # ② 转标准格式
  python fraud_pipeline.py profile  data.jsonl                # ③ 质量自检 + oracle 摸底
  python fraud_pipeline.py run      data.jsonl --shape operation -o outputs/     # ④ 训练+评估
  python fraud_pipeline.py score    新数据.jsonl --model outputs/model.pt        # ⑤ 给新数据打分

--shape 按数据形状选(见 ARCH_REPORT.md):
  operation   操作流(App日志/盗号场景): 自回归+时间偏置, top-k 池化
  funds       资金流(转账流水):          整事件遮罩, mean 池化
  consumption 消费流(卡交易):            字段级遮罩, top-k 池化

支持输入: .csv / .xlsx / .parquet / .jsonl(已是标准格式则跳过 convert)
标准格式: 一行一用户 {"user_id","label","events":[{"type","t","amount","result","channel","ip_change"}]}
依赖: torch pandas scikit-learn numpy
"""
import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np

SEED = 42
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_LBL = ["≤1m", "1-5m", "5-30m", "0.5-1h", "1-6h", "6-24h", "1-7d", ">7d", "首"]
RATIO_EDGES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_GAMT, N_PAMT, N_GAP = 8, len(RATIO_EDGES) + 2, len(GAP_EDGES) + 2


# ============================================================
# 通用读取
# ============================================================
def read_table(path):
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise SystemExit(f"不支持的表格格式: {ext} (支持 csv/xlsx/parquet; "
                     f"jsonl 已是标准格式无需 convert)")


# ============================================================
# ① inspect — 字段识别与映射模板
# ============================================================
ROLE_HINTS = {
    "user_col":   ["user", "账户", "账号", "客户", "卡号", "uid", "acct", "account", "cust"],
    "time_col":   ["time", "时间", "日期", "date", "ts", "timestamp"],
    "type_col":   ["type", "类型", "事件", "event", "操作", "category", "行为", "交易码"],
    "amount_col": ["amount", "金额", "amt", "money", "value", "价格"],
    "result_col": ["result", "结果", "状态", "status", "是否成功"],
    "channel_col": ["channel", "渠道", "来源", "端", "chip"],
    "ip_col":     ["ip"],
    "label_col":  ["label", "标签", "fraud", "欺诈", "黑", "是否欺诈", "isfraud", "风险"],
}


def guess_roles(df):
    guesses = {}
    for role, hints in ROLE_HINTS.items():
        for c in df.columns:
            cl = str(c).lower()
            if any(h in cl for h in hints):
                guesses[role] = c
                break
    return guesses


def cmd_inspect(args):
    df = read_table(args.file)
    print(f"\n{'='*64}\n字段清单 · {args.file} · {len(df)} 行 × {len(df.columns)} 列\n{'='*64}")
    print(f"{'列名':<20}{'类型':<10}{'非空率':>7}{'唯一值':>9}  示例")
    for c in df.columns:
        s = df[c]
        ex = str(s.dropna().iloc[0])[:24] if s.notna().any() else "—"
        print(f"{str(c):<20}{str(s.dtype):<10}{s.notna().mean():>6.0%}"
              f"{s.nunique():>9}  {ex}")
    g = guess_roles(df)
    print(f"\n自动猜测的字段角色(请人工确认!):")
    for role in ROLE_HINTS:
        print(f"  {role:<13} → {g.get(role, '(未识别, 请手工填或设 null)')}")
    mapping = {
        "user_col": g.get("user_col"), "time_col": g.get("time_col"),
        "type_col": g.get("type_col"), "amount_col": g.get("amount_col"),
        "result_col": g.get("result_col"), "channel_col": g.get("channel_col"),
        "ip_col": g.get("ip_col"), "label_col": g.get("label_col"),
        "success_values": ["成功", "S", "1", "success", "0000"],
        "time_format": None,
        "min_events": 10, "max_len": 200,
        "_说明": "确认/修改各列名; 没有的字段填 null; label_col 为空则全部当无标签处理",
    }
    out = args.out or "mapping.json"
    json.dump(mapping, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n映射模板已生成 → {out}")
    print(f"下一步: 确认 {out} 后执行\n  python fraud_pipeline.py convert "
          f"{args.file} -m {out} -o data.jsonl")


# ============================================================
# ② convert — 任意表 → 标准 jsonl
# ============================================================
def cmd_convert(args):
    import pandas as pd
    m = json.load(open(args.mapping, encoding="utf-8"))
    df = read_table(args.file)
    for k in ("user_col", "time_col", "type_col"):
        if not m.get(k) or m[k] not in df.columns:
            raise SystemExit(f"mapping 里 {k}={m.get(k)!r} 不存在于数据列中, 请修正")
    ts = pd.to_datetime(df[m["time_col"]], format=m.get("time_format"),
                        errors="coerce")
    bad = ts.isna().sum()
    if bad:
        print(f"警告: {bad} 行时间解析失败, 已丢弃")
    df = df[ts.notna()].copy()
    df["_ts"] = ts[ts.notna()]
    df.sort_values([m["user_col"], "_ts"], inplace=True, kind="mergesort")

    succ = set(str(v) for v in m.get("success_values", []))
    n_users = n_kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for uid, g in df.groupby(m["user_col"], sort=False):
            n_users += 1
            if len(g) < m.get("min_events", 10):
                continue
            g = g.tail(m.get("max_len", 200))
            events = []
            for _, r in g.iterrows():
                ev = {"type": str(r[m["type_col"]]),
                      "t": r["_ts"].strftime("%Y-%m-%d %H:%M:%S")}
                if m.get("amount_col") and pd.notna(r.get(m["amount_col"])):
                    try:
                        ev["amount"] = float(r[m["amount_col"]])
                    except (TypeError, ValueError):
                        pass
                if m.get("result_col") and pd.notna(r.get(m["result_col"])):
                    ev["result"] = ("成功" if str(r[m["result_col"]]) in succ
                                    else "失败")
                if m.get("channel_col") and pd.notna(r.get(m["channel_col"])):
                    ev["channel"] = str(r[m["channel_col"]])
                if m.get("ip_col") and pd.notna(r.get(m["ip_col"])):
                    try:
                        ev["ip_change"] = int(float(r[m["ip_col"]]))
                    except (TypeError, ValueError):
                        pass
                events.append(ev)
            label = 0
            if m.get("label_col") and m["label_col"] in g.columns:
                label = int(pd.to_numeric(g[m["label_col"]],
                                          errors="coerce").fillna(0).max() > 0)
            f.write(json.dumps({"user_id": str(uid), "label": label,
                                "events": events}, ensure_ascii=False) + "\n")
            n_kept += 1
    print(f"完成: {n_users} 个用户中保留 {n_kept} 个(≥{m.get('min_events',10)}事件)"
          f" → {args.out}")
    print(f"下一步:\n  python fraud_pipeline.py profile {args.out}")


# ============================================================
# 标准 jsonl 加载与编码
# ============================================================
def load_jsonl(path, max_len=200):
    users = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        evs = r["events"][:max_len]
        if len(evs) < 2:
            continue
        users.append({"uid": str(r["user_id"]), "label": int(r.get("label", 0)),
                      "events": evs})
    return users


class Encoder:
    """从训练用户构建词表/分位点; encode() 输出模型所需全部字段"""
    def __init__(self, train_users):
        from collections import Counter
        tc, cc = Counter(), Counter()
        amts = []
        has_res = has_ch = has_ip = False
        for u in train_users:
            for ev in u["events"]:
                tc[ev["type"]] += 1
                if "channel" in ev:
                    cc[ev["channel"]] += 1; has_ch = True
                if "result" in ev:
                    has_res = True
                if "ip_change" in ev:
                    has_ip = True
                a = ev.get("amount")
                if a and a > 0:
                    amts.append(a)
        self.type_vocab = {t: i for i, (t, _) in enumerate(tc.most_common(199))}
        self.ch_vocab = {c: i + 1 for i, (c, _) in
                         enumerate(cc.most_common(30))}
        self.gq = (np.quantile(np.array(amts),
                               np.linspace(0, 1, N_GAMT)[1:-1])
                   if len(amts) >= 50 else None)
        self.has_amount = self.gq is not None
        self.has_res, self.has_ch, self.has_ip = has_res, has_ch, has_ip
        self.n_types = len(self.type_vocab) + 1
        self.n_ch = len(self.ch_vocab) + 2

    def encode(self, u):
        evs = u["events"]
        nz = [ev["amount"] for ev in evs if ev.get("amount")]
        med = float(np.median(nz)) if nz else 1.0
        o = {"uid": u["uid"], "label": u["label"], "type": [], "res": [],
             "ch": [], "ip": [], "gamt": [], "pamt": [], "gap": [],
             "hour": [], "ts": []}
        prev = None
        for ev in evs:
            t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
            o["ts"].append(t.timestamp())
            o["type"].append(self.type_vocab.get(ev["type"],
                                                 self.n_types - 1))
            o["res"].append({"成功": 1, "失败": 2}.get(ev.get("result"), 0))
            o["ch"].append(self.ch_vocab.get(ev.get("channel"),
                                             0 if "channel" not in ev
                                             else self.n_ch - 1))
            o["ip"].append(0 if "ip_change" not in ev
                           else 1 + int(bool(ev["ip_change"])))
            a = ev.get("amount")
            if a and self.gq is not None:
                o["gamt"].append(1 + int(np.searchsorted(self.gq, a)))
                o["pamt"].append(1 + int(np.searchsorted(RATIO_EDGES, a / med)))
            else:
                o["gamt"].append(0); o["pamt"].append(0)
            o["gap"].append(N_GAP - 1 if prev is None else
                            int(np.searchsorted(GAP_EDGES,
                                                max((t - prev).total_seconds(), 0),
                                                side="left")))
            o["hour"].append(t.hour + t.minute / 60)
            prev = t
        return o

    def meta(self):
        return {"type_vocab": self.type_vocab, "ch_vocab": self.ch_vocab,
                "gq": None if self.gq is None else self.gq.tolist(),
                "has_amount": self.has_amount, "has_res": self.has_res,
                "has_ch": self.has_ch, "has_ip": self.has_ip,
                "n_types": self.n_types, "n_ch": self.n_ch}

    @classmethod
    def from_meta(cls, meta):
        enc = cls.__new__(cls)
        enc.type_vocab = meta["type_vocab"]
        enc.ch_vocab = meta["ch_vocab"]
        enc.gq = None if meta["gq"] is None else np.array(meta["gq"])
        enc.has_amount = meta["has_amount"]; enc.has_res = meta["has_res"]
        enc.has_ch = meta["has_ch"]; enc.has_ip = meta["has_ip"]
        enc.n_types = meta["n_types"]; enc.n_ch = meta["n_ch"]
        return enc


# ============================================================
# ③ profile — 质量自检 + oracle 摸底
# ============================================================
def cmd_profile(args):
    users = load_jsonl(args.file)
    w = [u for u in users if u["label"] == 0]
    b = [u for u in users if u["label"] == 1]
    print(f"\n{'='*64}\n数据概览 · {args.file}\n{'='*64}")
    print(f"用户: {len(users)} (白 {len(w)} / 黑 {len(b)})"
          + ("  ⚠️ 无黑标签, 只能出分数不能出 AUC" if len(b) == 0 else ""))
    lens = [len(u["events"]) for u in users]
    print(f"序列长度: p10={np.percentile(lens,10):.0f} "
          f"p50={np.median(lens):.0f} p90={np.percentile(lens,90):.0f}")

    def side_stats(us, name):
        from collections import Counter
        tc = Counter(); gaps = []; hours = []; amts = []
        cov = Counter()
        for u in us:
            prev = None
            for ev in u["events"]:
                tc[ev["type"]] += 1
                for k in ("amount", "result", "channel", "ip_change"):
                    if k in ev:
                        cov[k] += 1
                t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
                hours.append(t.hour)
                if prev:
                    gaps.append((t - prev).total_seconds())
                prev = t
                if ev.get("amount"):
                    amts.append(ev["amount"])
        n_ev = sum(tc.values())
        gh = np.histogram(np.searchsorted(GAP_EDGES, gaps, side="left"),
                          bins=range(9))[0] / max(len(gaps), 1)
        print(f"\n[{name}] 事件 {n_ev}, 类型 {len(tc)} 种, "
              f"top5: {', '.join(f'{k}({v})' for k,v in tc.most_common(5))}")
        print(f"  字段覆盖率: " + "  ".join(
            f"{k}={cov[k]/n_ev:.0%}" for k in
            ("amount", "result", "channel", "ip_change")))
        print(f"  间隔分布: " + " ".join(
            f"{GAP_LBL[i]}:{gh[i]:.0%}" for i in range(8) if gh[i] >= 0.01))
        if amts:
            print(f"  金额: p50={np.median(amts):,.0f} "
                  f"p90={np.percentile(amts,90):,.0f} max={max(amts):,.0f}")
        print(f"  平均小时: {np.mean(hours):.1f}")
        return tc

    tw = side_stats(w, "白样本")
    if b:
        tb = side_stats(b, "黑样本")
        # 词频差异提示
        keys = set(tw) | set(tb)
        nw, nb = sum(tw.values()), sum(tb.values())
        diffs = sorted(keys, key=lambda k: -abs(tw[k]/nw - tb[k]/nb))[:3]
        print(f"\n词频差异最大的类型: " + ", ".join(
            f"{k}(白{tw[k]/nw:.1%} vs 黑{tb[k]/nb:.1%})" for k in diffs))

        # oracle 摸底
        if len(w) >= 20 and len(b) >= 20:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
            enc = Encoder(w)
            eus = [enc.encode(u) for u in users]
            ya = np.array([u["label"] for u in eus])
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000))
            def cv(name, X):
                p = cross_val_predict(pipe, np.array(X), ya, cv=5,
                                      method="predict_proba")[:, 1]
                print(f"  {name:<22} AUC={roc_auc_score(ya, p):.4f}")
            print(f"\noracle 信号摸底(用标签, 5折):")
            nt = enc.n_types
            cv("词频", [[u['type'].count(k)/len(u['type'])
                        for k in range(min(nt, 50))] for u in eus])
            if enc.has_amount:
                cv("全局金额统计", [[max(u['gamt']), float(np.mean(u['gamt']))]
                                    for u in eus])
                cv("个人比值统计", [[max(u['pamt']),
                                    sum(1 for p in u['pamt'] if p == N_PAMT-1)]
                                    for u in eus])
            cv("时序密集统计", [[float(np.mean(np.array(u['gap']) <= 1)),
                                sum(1 for g_ in u['gap'] if g_ == 0)]
                                for u in eus])
            if enc.has_res:
                cv("失败次数统计", [[sum(1 for r_ in u['res'] if r_ == 2)]
                                    for u in eus])
            print("  读法: 哪层 AUC 高, 信号就在哪层; 全部≈0.5 说明数据无信号")
    print(f"\n下一步:\n  python fraud_pipeline.py run {args.file} "
          f"--shape operation -o outputs/")


# ============================================================
# ④ run / ⑤ score — 模型
# ============================================================
def build_model(shape, enc, d=64, layers=2):
    import torch
    import torch.nn as nn

    fields = ["type", "gap", "hour"]
    if enc.has_amount:
        fields += ["gamt", "pamt"]
    if enc.has_res:
        fields.append("res")
    if enc.has_ch:
        fields.append("ch")
    if enc.has_ip:
        fields.append("ip")
    n_cls = {"type": enc.n_types, "res": 3, "ch": enc.n_ch, "ip": 3,
             "gamt": N_GAMT, "pamt": N_PAMT, "gap": N_GAP}
    dims = {"type": 32, "res": 4, "ch": 8, "ip": 4, "gamt": 8,
            "pamt": 16, "gap": 16, "hour": 8}
    heads = ["type", "gap"] + (["pamt"] if enc.has_amount else []) \
            + (["res"] if enc.has_res else [])

    class TimeBiasLayer(nn.Module):
        def __init__(self, nhead=4, ff=128, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                              batch_first=True)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(ff, d))
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, bias, pad):
            a, _ = self.attn(x, x, x, attn_mask=bias, key_padding_mask=pad,
                             need_weights=False)
            x = self.n1(x + self.drop(a))
            return self.n2(x + self.drop(self.ff(x)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.shape, self.fields, self.heads_list = shape, fields, heads
            self.nhead = 4
            self.embs = nn.ModuleDict({"e_" + f: nn.Embedding(n_cls[f], dims[f])
                                       for f in fields if f != "hour"})
            self.hour_proj = nn.Linear(2, dims["hour"])
            in_dim = sum(dims[f] for f in fields)
            self.in_proj = nn.Linear(in_dim, d)
            self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
            self.fmask = nn.ParameterDict(
                {"m_" + f: nn.Parameter(torch.randn(dims[f]) * 0.02)
                 for f in fields if f != "hour"})
            self.pos_emb = nn.Embedding(512, d)
            self.time_bias = (shape == "operation")
            if self.time_bias:
                self.bias_emb = nn.Embedding(len(GAP_EDGES) + 1, self.nhead)
                self.layers = nn.ModuleList([TimeBiasLayer()
                                             for _ in range(layers)])
            else:
                lyr = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                                 dim_feedforward=128,
                                                 dropout=0.1, batch_first=True)
                self.encoder = nn.TransformerEncoder(lyr, num_layers=layers)
            self.heads = nn.ModuleDict({"h_" + h: nn.Linear(d, n_cls[h])
                                        for h in heads})

        def _embed(self, T, mask=None, mask_field=None):
            parts = []
            for f in fields:
                if f == "hour":
                    h = T["hour"]
                    parts.append(self.hour_proj(torch.stack(
                        [torch.sin(2 * math.pi * h / 24),
                         torch.cos(2 * math.pi * h / 24)], -1)))
                else:
                    e = self.embs["e_" + f](T[f])
                    if mask is not None and mask_field is not None:
                        blocked = ({"gamt", "pamt"} if mask_field in
                                   ("gamt", "pamt") else {mask_field})
                        if f in blocked:
                            e = torch.where(mask.unsqueeze(-1),
                                            self.fmask["m_" + f].expand_as(e), e)
                    parts.append(e)
            x = self.in_proj(torch.cat(parts, -1))
            if mask is not None and mask_field is None:      # 整事件遮罩
                x = torch.where(mask.unsqueeze(-1),
                                self.mask_emb.expand_as(x), x)
            L = x.size(1)
            x = x + self.pos_emb(torch.arange(L, device=x.device)
                                 .clamp(max=511))[None]
            return x

        def forward(self, T, pad, mask=None, mask_field=None):
            x = self._embed(T, mask, mask_field)
            B, L = pad.shape
            bias = None
            if self.time_bias:
                dt = (T["ts"][:, :, None] - T["ts"][:, None, :]).abs()
                dtb = torch.bucketize(dt.float(), torch.tensor(
                    GAP_EDGES, dtype=torch.float32, device=dt.device))
                bias = self.bias_emb(dtb).permute(0, 3, 1, 2).reshape(
                    B * self.nhead, L, L)
            if shape == "operation":                          # 因果
                cm = torch.triu(torch.full((L, L), float("-inf"),
                                           device=x.device), 1)
                bias = cm if bias is None else bias + cm[None]
            if self.time_bias:
                for lyr in self.layers:
                    x = lyr(x, bias, pad)
                h = x
            else:
                h = self.encoder(x, mask=bias,
                                 src_key_padding_mask=pad)
            return {k: hd(h) for k, hd in self.heads.items()}

    return Model()


def to_tensors(batch, device, need_ts):
    import torch
    n = len(batch)
    L = max(len(u["type"]) for u in batch)
    T = {}
    for f in ("type", "res", "ch", "ip", "gamt", "pamt", "gap"):
        M = torch.zeros(n, L, dtype=torch.long)
        for i, u in enumerate(batch):
            M[i, :len(u[f])] = torch.tensor(u[f])
        T[f] = M.to(device)
    H = torch.zeros(n, L)
    pad = torch.ones(n, L, dtype=torch.bool)
    for i, u in enumerate(batch):
        m = len(u["hour"])
        H[i, :m] = torch.tensor(u["hour"])
        pad[i, :m] = False
    T["hour"] = H.to(device)
    if need_ts:
        TS = torch.zeros(n, L, dtype=torch.float64)
        for i, u in enumerate(batch):
            TS[i, :len(u["ts"])] = torch.tensor(u["ts"], dtype=torch.float64)
        T["ts"] = TS.to(device)
    return T, pad.to(device)


def train_model(model, users, shape, device, epochs, bs=32, lr=1e-3):
    import torch
    import torch.nn as nn
    ce = nn.functional.cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(SEED)
    hm = {("h_" + h): h for h in model.heads_list}
    fm_fields = [f for f in ("type", "gamt", "gap")
                 if f in model.fields]                       # 消费流轮换遮的字段
    model.train()
    for ep in range(epochs):
        order = list(range(len(users)))
        random.shuffle(order)
        tot = nb = 0
        for s in range(0, len(order), bs):
            batch = [users[j] for j in order[s:s + bs]]
            T, pad = to_tensors(batch, device, model.time_bias)
            if shape == "operation":                          # 自回归
                out = model(T, pad)
                m = ~pad[:, 1:]
                loss = sum(ce(out[k][:, :-1][m], T[f][:, 1:][m])
                           for k, f in hm.items())
            elif shape == "funds":                            # 整事件遮罩
                sc = torch.rand(pad.shape, generator=gen); sc[pad] = -1
                mask = (sc > 0.85).to(device)
                mask[:, 0] |= ~mask.any(1)
                out = model(T, pad, mask=mask)
                loss = sum(ce(out[k][mask], T[f][mask]) for k, f in hm.items())
            else:                                             # 字段级遮罩
                fld = fm_fields[nb % len(fm_fields)]
                sc = torch.rand(pad.shape, generator=gen); sc[pad] = -1
                mask = (sc > 0.85).to(device)
                mask[:, 0] |= ~mask.any(1)
                hk = "h_" + ("pamt" if fld == "gamt" and
                             "pamt" in model.heads_list else fld)
                tf = hm.get(hk, fld)
                out = model(T, pad, mask=mask, mask_field=fld)
                if hk in out:
                    loss = ce(out[hk][mask], T[tf][mask])
                else:
                    loss = ce(out["h_type"][mask], T["type"][mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  loss {tot/nb:.4f}", flush=True)


def per_position_ce(model, users, shape, device, bs=64, stride=7):
    import torch
    import torch.nn as nn
    ce = nn.functional.cross_entropy
    model.eval()
    hm = {("h_" + h): h for h in model.heads_list}
    out = [{h: np.zeros(len(u["type"])) for h in model.heads_list}
           for u in users]
    for i, u in enumerate(users):
        out[i]["gb"] = np.array(u["gap"])
    with torch.no_grad():
        if shape == "operation":                              # 单趟
            for s in range(0, len(users), bs):
                batch = users[s:s + bs]
                T, pad = to_tensors(batch, device, model.time_bias)
                o = model(T, pad)
                for i in range(len(batch)):
                    m = len(batch[i]["type"])
                    if m < 2:
                        continue
                    for k, f in hm.items():
                        out[s+i][f][1:m] = ce(o[k][i, :m-1], T[f][i, 1:m],
                                              reduction="none").cpu().numpy()
        else:                                                 # 步长遮罩
            for r in range(stride):
                for s in range(0, len(users), bs):
                    batch = users[s:s + bs]
                    T, pad = to_tensors(batch, device, model.time_bias)
                    pos = torch.arange(T["type"].size(1))
                    mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
                    if not mask.any():
                        continue
                    o = model(T, pad, mask=mask.to(device))
                    for i in range(len(batch)):
                        idx = mask[i].nonzero().flatten()
                        if not len(idx):
                            continue
                        for k, f in hm.items():
                            out[s+i][f][idx.numpy()] = ce(
                                o[k][i, idx], T[f][i, idx],
                                reduction="none").cpu().numpy()
    return out


def norm_stats(pcs_tr, key):
    """训练白样本按间隔桶统计 CE 均值/方差 —— 保存进 checkpoint 供 score 复用"""
    av = np.concatenate([p[key] for p in pcs_tr])
    ab = np.concatenate([p["gb"] for p in pcs_tr])
    mu = np.full(N_GAP, av.mean()); sd = np.full(N_GAP, max(av.std(), 1e-3))
    for b_ in range(N_GAP):
        m = ab == b_
        if m.sum() >= 30:
            mu[b_], sd[b_] = av[m].mean(), max(av[m].std(), 1e-3)
    return mu, sd


def apply_norm(pcs, key, mu, sd):
    return [(p[key] - mu[np.clip(p["gb"], 0, N_GAP-1)]) /
            sd[np.clip(p["gb"], 0, N_GAP-1)] for p in pcs]


def z_norm(pcs_tr, pcs, key):
    mu, sd = norm_stats(pcs_tr, key)
    return apply_norm(pcs, key, mu, sd)


def pool_scores(zs, shape, heads):
    combos = {h: zs[h] for h in heads}
    combos["SUM"] = [sum(v) for v in zip(*[zs[h] for h in heads])]
    aggs = ([("mean", lambda a: float(a.mean()))] if shape == "funds" else []) \
        + [(f"top{k}", (lambda kk: lambda a:
            float(np.sort(a)[-min(kk, len(a)):].mean()))(k)) for k in (3, 5, 10)]
    return combos, aggs



# ============================================================
# ⑥ diagnose — 六关卡全链路诊断: 分不出来时自动定位原因
# ============================================================
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def cmd_diagnose(args):
    import torch
    from collections import Counter
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    R = []          # 报告行
    verdicts = []   # (关卡, 状态, 说明)

    def log(line=""):
        print(line, flush=True); R.append(line)

    def verdict(gate, status, msg):
        verdicts.append((gate, status, msg))
        log(f"  [{status}] {msg}")

    users = load_jsonl(args.file)
    w = [u for u in users if u["label"] == 0]
    b = [u for u in users if u["label"] == 1]
    has_labels = len(b) >= 10

    # ---------------- 关卡 C1: 数据体检 ----------------
    log("=" * 60); log("关卡 C1 · 数据体检 (原材料有没有问题)"); log("=" * 60)
    log(f"  用户 {len(users)} (白 {len(w)} / 黑 {len(b)}), "
        f"序列长度中位 {int(np.median([len(u['events']) for u in users]))}")
    gaps_all, starts, sec_zero, n_ts = [], Counter(), 0, 0
    for u in users:
        prev = None
        first = True
        for ev in u["events"]:
            t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
            n_ts += 1
            sec_zero += (t.second == 0)
            if first:
                starts[t.strftime("%H:%M")] += 1; first = False
            if prev:
                gaps_all.append(
                    int(np.searchsorted(GAP_EDGES,
                                        max((t - prev).total_seconds(), 0),
                                        side="left")))
            prev = t
    gh = np.bincount(gaps_all, minlength=8)[:8] / max(len(gaps_all), 1)
    log("  间隔桶分布: " + " ".join(f"{GAP_LBL[i]}:{gh[i]:.0%}"
                                    for i in range(8) if gh[i] >= 0.01))
    if gh.max() > 0.8:
        verdict("C1", "❌", f"时间粒度退化: {gh.argmax()}号桶占{gh.max():.0%}"
                " —— 间隔字段近乎常数, 时序信号基本不可用, 检查时间戳精度/分桶边界")
    else:
        verdict("C1", "✅", "间隔分布有多样性")
    top_start, top_n = starts.most_common(1)[0]
    if top_n / len(users) > 0.3:
        verdict("C1", "⚠️", f"起始时间高度集中: {top_n/len(users):.0%} 的用户从 "
                f"{top_start} 开始 —— 疑似采集/生成伪影, 可能成为泄漏捷径")
    if sec_zero / n_ts > 0.95:
        verdict("C1", "⚠️", "时间戳几乎全为整分钟 —— 秒级信息缺失, 密集爆发类信号会被钝化")
    if not has_labels:
        verdict("C1", "⚠️", f"黑样本仅 {len(b)} 个(<10) —— 无法计算 AUC, "
                "后续关卡只能给部分诊断")

    # ---------------- 关卡 C2: 信号存在性 (oracle 分层摸底) ----------------
    log(""); log("=" * 60)
    log("关卡 C2 · 信号存在性 (数据里到底有没有可分的信号)"); log("=" * 60)
    rng = random.Random(SEED); rng.shuffle(w)
    n_tr = int(len(w) * 0.8)
    train_w, test_w = w[:n_tr], w[n_tr:]
    enc = Encoder(train_w)
    etr = [enc.encode(u) for u in train_w]
    ete = [enc.encode(u) for u in test_w + b]
    y = np.array([u["label"] for u in ete])
    oracle_layers = {}
    if has_labels:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        allu = etr + ete
        ya = np.array([u["label"] for u in allu])
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000))
        def cv(name, X):
            p = cross_val_predict(pipe, np.array(X), ya, cv=5,
                                  method="predict_proba")[:, 1]
            a = _auc(ya, p); oracle_layers[name] = a
            log(f"  oracle[{name}] AUC = {a:.4f}")
        cv("词频/模式层", [[u["type"].count(k)/len(u["type"])
                           for k in range(min(enc.n_types, 50))] for u in allu])
        if enc.has_amount:
            cv("个人金额层", [[max(u["pamt"]),
                              sum(1 for p_ in u["pamt"] if p_ == N_PAMT-1)]
                             for u in allu])
        cv("时序密集层", [[float(np.mean(np.array(u["gap"]) <= 1)),
                          sum(1 for g_ in u["gap"] if g_ == 0)] for u in allu])
        if enc.has_res:
            cv("状态/失败层", [[sum(1 for r_ in u["res"] if r_ == 2)]
                              for u in allu])
        omax = max(oracle_layers.values())
        obest = max(oracle_layers, key=oracle_layers.get)
        if omax < 0.58:
            verdict("C2", "❌", f"所有信号层 oracle 都≈随机(最高 {omax:.3f}) —— "
                    "根因大概率在这里: 数据里没有可分信号。任何模型都救不了, "
                    "回头核查: 黑样本定义是否正确/关键字段是否缺失/欺诈是否根本不在行为层")
        elif omax < 0.7:
            verdict("C2", "⚠️", f"信号偏弱(最强层 {obest}={omax:.3f}) —— "
                    "可分但上限不高, 模型预期 ≈ 上限−0.03~0.07")
        else:
            verdict("C2", "✅", f"信号存在, 最强层={obest}({omax:.3f}), "
                    f"模型预期落点 {omax-0.07:.2f}~{omax-0.03:.2f}")
    else:
        log("  (无标签, 跳过 oracle)")

    # ---------------- 关卡 C3: 训练体检 (模型学到语法了吗) ----------------
    log(""); log("=" * 60)
    log("关卡 C3 · 训练体检 (模型有没有学到'正常语法')"); log("=" * 60)
    n_val = max(20, len(etr) // 10)
    etr_fit, etr_val = etr[:-n_val], etr[-n_val:]
    model = build_model(args.shape, enc)
    train_model(model, etr_fit, args.shape, "cpu", epochs=args.epochs)
    pcs_val = per_position_ce(model, etr_val, args.shape, "cpu")
    n_cls = {"type": enc.n_types, "gap": N_GAP, "pamt": N_PAMT, "res": 3}
    log("  留出白样本上的预测能力(语法学习度 = 1 − 实际CE/瞎猜CE):")
    grammar = {}
    for h in model.heads_list:
        ce_val = float(np.mean(np.concatenate([p[h] for p in pcs_val])))
        base = math.log(n_cls[h])
        g = max(0.0, 1 - ce_val / base)
        grammar[h] = g
        log(f"    {h:<6} CE={ce_val:.3f} vs 瞎猜{base:.3f} → 语法学习度 {g:.0%}")
    if max(grammar.values()) < 0.10:
        verdict("C3", "❌", "模型在白样本上几乎不比瞎猜强 —— 白样本行为本身接近随机, "
                "没有可学的'正常语法'(生成/采集问题), 惊讶度打分将失效")
    else:
        gb_ = max(grammar, key=grammar.get)
        verdict("C3", "✅", f"学到语法(最强头 {gb_} 学习度 {grammar[gb_]:.0%})")

    # ---------------- 关卡 C4: 头级分离度 ----------------
    log(""); log("=" * 60)
    log("关卡 C4 · 头级分离度 (哪路信号在起作用/失效)"); log("=" * 60)
    pcs_tr = per_position_ce(model, etr_fit, args.shape, "cpu")
    pcs = per_position_ce(model, ete, args.shape, "cpu")
    zs = {h: z_norm(pcs_tr, pcs, h) for h in model.heads_list}
    head_auc = {}
    if has_labels:
        for h in model.heads_list:
            s5 = np.array([float(np.sort(a)[-min(5, len(a)):].mean())
                           for a in zs[h]])
            sm = np.array([float(a.mean()) for a in zs[h]])
            head_auc[h] = max(_auc(y, s5), _auc(y, sm))
            log(f"  头[{h:<5}] AUC = {head_auc[h]:.4f}"
                + ("  ← 反向!" if head_auc[h] < 0.45 else ""))
        dead = [h for h, a in head_auc.items() if 0.45 <= a < 0.55]
        if dead:
            verdict("C4", "⚠️", f"无信号的头: {','.join(dead)} —— 这些字段在该数据上"
                    "不携带判别信息(与 C2 各层对照可知信号丢在编码还是数据)")
        inv = [h for h, a in head_auc.items() if a < 0.45]
        if inv:
            verdict("C4", "❌", f"反向的头: {','.join(inv)} —— 疑似熵混淆残留或"
                    "该字段黑样本反而更规律, 单头剔除或检查归一化分桶")
        if any(a >= 0.6 for a in head_auc.values()):
            hb = max(head_auc, key=head_auc.get)
            verdict("C4", "✅", f"有效头存在(最强 {hb}={head_auc[hb]:.3f})")

    # ---------------- 关卡 C5: 打分方式诊断 ----------------
    log(""); log("=" * 60)
    log("关卡 C5 · 打分方式诊断 (归一化/池化选对了吗)"); log("=" * 60)
    combos, aggs = pool_scores(zs, args.shape, model.heads_list)
    best = (0.5, None, None)
    if has_labels:
        raw_mean = np.array([float(np.mean(sum(p[h] for h in model.heads_list)))
                             for p in pcs])
        a_raw = _auc(y, raw_mean)
        log(f"  raw平均(不归一化) AUC = {a_raw:.4f}")
        for cn, z in combos.items():
            for an, fn in aggs:
                s = np.array([fn(a) for a in z])
                a_ = _auc(y, s)
                if a_ > best[0]:
                    best = (a_, f"{cn}|{an}", s)
        log(f"  z-norm 最优组合   AUC = {best[0]:.4f}  ({best[1]})")
        zm = {an: max(_auc(y, np.array([fn(a) for a in combos['SUM']])), 0)
              for an, fn in aggs}
        log("  池化对比(SUM头): " + "  ".join(f"{k}={v:.3f}"
                                              for k, v in zm.items()))
        if a_raw < 0.45 and best[0] > 0.6:
            verdict("C5", "✅", "检测到熵混淆且已被 z-norm 修复(raw反向→z-norm正常), 属预期行为")
        if best[0] - a_raw > 0.1:
            verdict("C5", "✅", f"归一化贡献显著(+{best[0]-a_raw:.2f})")
        mean_a = zm.get("mean")
        topk_a = max((v for k, v in zm.items() if k.startswith("top")),
                     default=None)
        # 只有 mean 与 top-k 两种池化都存在时才做对比, 避免单边误报
        if mean_a is not None and topk_a is not None \
                and abs(mean_a - topk_a) > 0.05:
            better = "mean(整体型异常)" if mean_a > topk_a else "top-k(局部型异常)"
            verdict("C5", "✅", f"池化方式敏感, 该数据适合 {better}")

    # ---------------- 关卡 C6: 根因判定 ----------------
    log(""); log("=" * 60)
    log("关卡 C6 · 根因判定"); log("=" * 60)
    if has_labels:
        final_auc = best[0]
        log(f"  最终无监督 AUC = {final_auc:.4f}")
        if oracle_layers:
            omax = max(oracle_layers.values())
            obest = max(oracle_layers, key=oracle_layers.get)
            gap_o = omax - final_auc
            log(f"  oracle 上限 = {omax:.4f} ({obest}), 差距 = {gap_o:+.4f}")
            if omax < 0.58:
                log("  ➤ 根因: 【数据无信号】所有层 oracle≈随机。行动: 回查黑样本定义、"
                    "补字段(对手方/设备/结果)、或信号在图层(团伙)需图方法")
            elif final_auc >= omax - 0.08:
                log("  ➤ 结论: 【正常发挥】已达数据上限的 92%+。想再提升只能给数据"
                    "补信息(字段优先级: 个人基线/result/对手方新旧), 不必再调模型")
            elif max(grammar.values()) < 0.10:
                log("  ➤ 根因: 【白样本无语法】模型学不到正常规律(C3 ❌)。"
                    "行动: 核查白样本是否真实用户行为、采集是否失真")
            else:
                gaps_expl = []
                if oracle_layers.get("个人金额层", 0) > final_auc + 0.05 or \
                   oracle_layers.get("状态/失败层", 0) > final_auc + 0.05:
                    gaps_expl.append("某 oracle 层明显高于模型 → 该层信号是聚合型, "
                                     "改走分数融合通道(勿塞进模型)")
                if len(b) / max(len(users), 1) > 0.15:
                    gaps_expl.append(f"黑样本占比 {len(b)/len(users):.0%}>15% → "
                                     "训练污染超验证上限, 需少量标签预过滤训练集")
                if not gaps_expl:
                    gaps_expl.append("差距超预期但无明显单一原因 → 依次尝试: "
                                     "换 shape 配置 / 加训练轮数 / 检查间隔分桶边界")
                for e_ in gaps_expl:
                    log(f"  ➤ 可能根因: {e_}")
    else:
        log("  无标签模式: 已输出 C1/C3 体检结论。拿分数 top 榜单人工审核 1 周, "
            "审核结果即是评估答案")
    log(""); log("=" * 60); log("诊断汇总"); log("=" * 60)
    for g, s, m in verdicts:
        log(f"  {g} [{s}] {m}")
    path = os.path.join(args.out, "diagnosis.txt")
    open(path, "w", encoding="utf-8").write("\n".join(R))
    log(f"\n完整诊断 → {path}")


def cmd_run(args):
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    users = load_jsonl(args.file)
    w = [u for u in users if u["label"] == 0]
    b = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(w)
    n_tr = int(len(w) * 0.8)
    train_w, test_w = w[:n_tr], w[n_tr:]
    print(f"训练 {len(train_w)} 白 | 测试 {len(test_w)} 白 + {len(b)} 黑 | "
          f"形状={args.shape}")
    enc = Encoder(train_w)
    print(f"字段: 类型{enc.n_types-1}种 金额{'有' if enc.has_amount else '无'} "
          f"result{'有' if enc.has_res else '无'} "
          f"channel{'有' if enc.has_ch else '无'} "
          f"ip{'有' if enc.has_ip else '无'}")
    etr = [enc.encode(u) for u in train_w]
    ete = [enc.encode(u) for u in test_w + b]
    y = np.array([u["label"] for u in ete])

    model = build_model(args.shape, enc)
    print(f"模型: {sum(p.numel() for p in model.parameters())/1e3:.0f}K 参数")
    train_model(model, etr, args.shape, "cpu", epochs=args.epochs)

    print("打分中...", flush=True)
    pcs_tr = per_position_ce(model, etr, args.shape, "cpu")
    pcs = per_position_ce(model, ete, args.shape, "cpu")
    zs = {h: z_norm(pcs_tr, pcs, h) for h in model.heads_list}
    combos, aggs = pool_scores(zs, args.shape, model.heads_list)

    from sklearn.metrics import roc_auc_score, roc_curve
    report = [f"数据={args.file} 形状={args.shape} "
              f"训练白={len(train_w)} 测试={len(test_w)}白+{len(b)}黑"]
    best = (None, -1, None, None)
    for cn, z in combos.items():
        for an, fn in aggs:
            s = np.array([fn(a) for a in z])
            if len(b) >= 10:
                auc = roc_auc_score(y, s)
                fpr, tpr, _ = roc_curve(y, s)
                r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
                line = (f"{cn} {an:<6} AUC={auc:.4f} "
                        f"KS={np.max(tpr-fpr):.4f} R@FPR1%={r1:.1%}")
                if auc > best[1]:
                    best = (f"{cn}|{an}", auc, s, z)
            else:
                line = f"{cn} {an:<6} (黑样本不足, 仅输出分数)"
                if best[0] is None:
                    best = (f"{cn}|{an}", 0, np.array([fn(a) for a in z]), z)
            print("  " + line); report.append(line)

    cn, an = best[0].split("|")
    s = best[2]
    thr = float(np.quantile(s[y == 0], 0.99)) if (y == 0).any() else float("nan")
    summary = (f"\n最优打分: {best[0]}"
               + (f"  AUC={best[1]:.4f}" if len(b) >= 10 else "")
               + f"\n阈值@误报1% = {thr:.3f} (白样本p99)")
    print(summary); report.append(summary)

    # 保存(含训练期归一化统计, score 时复用保证阈值口径一致)
    nstats = {h: [a.tolist() for a in norm_stats(pcs_tr, h)]
              for h in model.heads_list}
    torch.save({"state": model.state_dict(), "meta": enc.meta(),
                "shape": args.shape, "best": best[0], "thr": thr,
                "norm_stats": nstats},
               os.path.join(args.out, "model.pt"))
    import csv
    with open(os.path.join(args.out, "scores.csv"), "w", newline="",
              encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["user_id", "label", "score", "命中(超阈值)",
                     "top3可疑位置(事件序号)"])
        for i, u in enumerate(ete):
            top3 = np.argsort(-best[3][i])[:3]     # 该用户 z 分最高的 3 个位置
            cw.writerow([u["uid"], u["label"], f"{s[i]:.4f}",
                         int(s[i] >= thr), " ".join(map(str, top3))])
    open(os.path.join(args.out, "report.txt"), "w",
         encoding="utf-8").write("\n".join(report))
    print(f"\n产出 → {args.out}/model.pt  scores.csv  report.txt")


def cmd_score(args):
    import torch
    ckpt = torch.load(args.model)
    enc = Encoder.from_meta(ckpt["meta"])
    shape = ckpt["shape"]
    users = load_jsonl(args.file)
    eus = [enc.encode(u) for u in users]
    model = build_model(shape, enc)
    model.load_state_dict(ckpt["state"])
    print(f"给 {len(users)} 个用户打分 (形状={shape}, "
          f"最优口径={ckpt['best']}, 阈值={ckpt['thr']:.3f})")
    pcs = per_position_ce(model, eus, shape, "cpu")
    ns = ckpt["norm_stats"]           # 复用训练期归一化统计, 口径与阈值一致
    zs = {h: apply_norm(pcs, h, np.array(ns[h][0]), np.array(ns[h][1]))
          for h in model.heads_list}
    combos, aggs = pool_scores(zs, shape, model.heads_list)
    cn, an = ckpt["best"].split("|")
    fn = dict(aggs)[an]
    s = np.array([fn(a) for a in combos[cn]])
    import csv
    out = args.out or "new_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["user_id", "score", "命中(超训练期阈值)"])
        for i, u in enumerate(eus):
            cw.writerow([u["uid"], f"{s[i]:.4f}",
                         int(s[i] >= ckpt["thr"])])
    print(f"完成 → {out} (命中 {int((s >= ckpt['thr']).sum())}/{len(s)})")


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="行为序列反欺诈统一流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="① 查看原始表字段, 生成映射模板")
    p.add_argument("file"); p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("convert", help="② 原始表 → 标准 jsonl")
    p.add_argument("file"); p.add_argument("-m", "--mapping", required=True)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_convert)
    p = sub.add_parser("profile", help="③ 质量自检 + oracle 摸底")
    p.add_argument("file"); p.set_defaults(fn=cmd_profile)
    p = sub.add_parser("diagnose", help="⑥ 全链路六关卡诊断: 分不出来时定位原因")
    p.add_argument("file")
    p.add_argument("--shape", choices=["operation", "funds", "consumption"],
                   required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("-o", "--out", default="outputs")
    p.set_defaults(fn=cmd_diagnose)
    p = sub.add_parser("run", help="④ 训练 + 评估")
    p.add_argument("file")
    p.add_argument("--shape", choices=["operation", "funds", "consumption"],
                   required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("-o", "--out", default="outputs")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("score", help="⑤ 用已训模型给新数据打分")
    p.add_argument("file"); p.add_argument("--model", required=True)
    p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
