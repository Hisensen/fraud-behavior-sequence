# -*- coding: utf-8 -*-
"""
反欺诈一体化系统 Web 应用 (本地, 无外部依赖, stdlib http.server)

  python webapp.py serve [端口=8765]     # 起服务, 浏览器打开 http://127.0.0.1:8765
  python webapp.py train 数据.jsonl 模型名 [epochs=30]   # 命令行直接训练

训练: 只用白样本 → MEM + iForest + 近邻库 + 人群聚类 + (有黑标签时)案件库
     → 全部打包存 models/<名字>.pkl
测试: 选模型 + 选账户(或粘贴一条 JSON) → 三栏体检单
     (可疑吗/哪类人/像什么案) + 逐笔定位
"""
import io
import json
import os
import pickle
import random
import sys
import threading
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

import mem_rich as M
from cluster_experiment import embed, surprise_profile

os.environ.setdefault("OMP_NUM_THREADS", "1")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
MODELS_DIR = MODELS
os.makedirs(MODELS, exist_ok=True)


# ---------------- 特征 / 编码 ----------------
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


def encode_rec(r, gq):
    """单条记录 → mem_rich 用户 dict (复刻 M.load 内循环)。"""
    evs = r["events"][:M.MAX_LEN]
    amts = [ev.get("amount") for ev in evs]
    nz = [a for a in amts if a]
    med = np.median(nz) if nz else 1.0
    u = {k: [] for k in ("type", "res", "ch", "ip", "gamt", "pamt",
                          "gap", "hour")}
    u["ts"] = []
    prev = None
    for ev in evs:
        t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
        u["ts"].append(t.timestamp())
        u["type"].append(M.T2I[ev["type"]])
        u["res"].append({"成功": 1, "失败": 2}.get(ev.get("result"), 0))
        u["ch"].append(M.CH2I.get(ev.get("channel"), 0))
        u["ip"].append(0 if "ip_change" not in ev else 1 + ev["ip_change"])
        a = ev.get("amount")
        u["gamt"].append(0 if not a else 1 + int(np.searchsorted(gq, a)))
        u["pamt"].append(0 if not a else
                         1 + int(np.searchsorted(M.RATIO_EDGES, a / med)))
        u["gap"].append(M.GAP_BOS if prev is None else
                        int(np.searchsorted(M.GAP_EDGES,
                                            max((t - prev).total_seconds(), 0),
                                            side="left")))
        u["hour"].append(t.hour + t.minute / 60)
        prev = t
    u["uid"] = r.get("user_id", "?")
    u["label"] = r.get("label", -1)
    return u


def znorm_stats(pcs_tr):
    """训练白样本 → 每头每间隔桶的 (mu, sd)。"""
    out = {}
    for k in pcs_tr[0]:
        if k == "gb":
            continue
        av = np.concatenate([p[k] for p in pcs_tr])
        ab = np.concatenate([p["gb"] for p in pcs_tr])
        mu = np.full(M.N_GAP, av.mean()); sd = np.full(M.N_GAP, max(av.std(), 1e-3))
        for b in range(M.N_GAP):
            m = ab == b
            if m.sum() >= 30:
                mu[b], sd[b] = av[m].mean(), max(av[m].std(), 1e-3)
        out[k] = (mu, sd)
    return out


def apply_znorm(pcs, zn):
    zs = {}
    for k, (mu, sd) in zn.items():
        zs[k] = [(p[k] - mu[np.clip(p["gb"], 0, M.N_GAP - 1)]) /
                 sd[np.clip(p["gb"], 0, M.N_GAP - 1)] for p in pcs]
    return zs


def pct_among(score, ref):
    return float((score > ref).mean() * 100)


# ---------------- 训练 ----------------
def train_bundle(data_path, name, epochs, log):
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    log(f"读取 {data_path} …")
    gq = M.global_quantiles(data_path)
    raw = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    users = [encode_rec(r, gq) for r in raw]
    meta = {r["user_id"]: r for r in raw}
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED); rng.shuffle(whites)
    train_w = whites[:int(len(whites) * 0.8)]
    log(f"共 {len(users)} 账户 | 训练白 {len(train_w)} | 黑(仅用于案件库) {len(blacks)}")

    log(f"① 训练 MEM ({epochs} epochs, 只用白样本)…")
    fields = ["type", "gamt", "gap", "hour", "res", "ch", "ip", "pamt"]
    model = M.MEMRich(fields)
    _print = print
    import builtins
    builtins.print = lambda *a, **k: log(" ".join(str(x) for x in a))
    try:
        M.train(model, train_w, "cpu", epochs=epochs)
    finally:
        builtins.print = _print
    log("① 计算白样本惊讶度基线…")
    pcs_tr = M.per_position_ce(model, train_w, "cpu")
    zn = znorm_stats(pcs_tr)
    zs_tr = apply_znorm(pcs_tr, zn)
    zsum_tr = [sum(v) for v in zip(*zs_tr.values())]
    c1m_w = np.array([float(a.mean()) for a in zsum_tr])
    c1t_w = np.array([M.topk_mean(a, 5) for a in zsum_tr])

    log("② 统计特征 + iForest…")
    Sw = np.array([stats_feats(u) for u in train_w])
    st_sc = StandardScaler().fit(Sw)
    Sws = st_sc.transform(Sw)
    ifo = IsolationForest(n_estimators=300, random_state=SEED).fit(Sws)
    c2_w = -ifo.score_samples(Sws)
    log("④a 近邻白样本库…")
    nn5 = NearestNeighbors(n_neighbors=6).fit(Sws)   # 白样本自查跳过自己
    c4_w = nn5.kneighbors(Sws)[0][:, -1]

    log("⑤ 人群聚类(簇数轮廓系数自选)…")
    from sklearn.metrics import silhouette_score
    Ew = embed(model, train_w, "cpu")
    emb_sc = StandardScaler().fit(Ew)
    Es = emb_sc.transform(Ew)
    best = None
    for kk in range(2, 9):
        km_try = KMeans(kk, n_init=10, random_state=SEED).fit(Es)
        s = silhouette_score(Es, km_try.labels_)
        log(f"  k={kk} 轮廓系数 {s:.3f}")
        if best is None or s > best[1]:
            best = (kk, s, km_try)
    km = best[2]
    log(f"  → 自选人群簇数 k={best[0]}")
    profiles = []
    for k in range(km.n_clusters):
        idx = [i for i, l in enumerate(km.labels_) if l == k]
        us = [train_w[i] for i in idx]
        maj = f"簇{k}"          # 不偷看真值; 名字由第2步 LLM 定类给出
        ch = [c for u in us for c in u["ch"] if c]
        night = float(np.mean([sum(1 for h in u["hour"] if h < 6) /
                               len(u["hour"]) for u in us]))
        profiles.append({"k": k, "n": len(idx), "name": maj,
                         "top_channel": {1: "APP", 2: "WEB", 3: "POS"}
                         [max(set(ch), key=ch.count)] if ch else "-",
                         "night": night})
    log("  人群: " + " / ".join(f"簇{p['k']}={p['name']}({p['n']})"
                                for p in profiles))

    caselib = None
    if blacks:
        log(f"④b 案件库({len(blacks)} 个带标签黑样本)…")
        pcs_b = M.per_position_ce(model, blacks, "cpu")
        zs_b = apply_znorm(pcs_b, zn)
        fp_b = surprise_profile(blacks, zs_b)
        fp_w = surprise_profile(train_w, zs_tr)
        fp_all = np.vstack([fp_w, fp_b])
        fp_sc = StandardScaler().fit(fp_all)
        caselib = {"scaler": fp_sc,
                   "X": fp_sc.transform(fp_all),
                   "is_black": np.array([False] * len(train_w) +
                                        [True] * len(blacks)),
                   "uid": [u["uid"] for u in train_w] +
                          [u["uid"] for u in blacks],
                   "btype": [""] * len(train_w) +
                            [meta[u["uid"]].get("btype") or "未知手法"
                             for u in blacks]}

    # 白样本融合分布 → 阈值
    fused_w = np.array([np.mean([pct_among(c1m_w[i], c1m_w),
                                 pct_among(c2_w[i], c2_w),
                                 pct_among(c4_w[i], c4_w)])
                        for i in range(len(train_w))])
    thr = float(np.percentile(fused_w, 99))
    log(f"阈值: 白样本融合分 p99 = {thr:.1f}")

    bundle = {"name": name, "data": os.path.basename(data_path),
              "gq": gq, "state": model.state_dict(), "fields": fields,
              "zn": zn, "c1m_w": c1m_w, "c1t_w": c1t_w, "c2_w": c2_w,
              "c4_w": c4_w, "st_sc": st_sc, "ifo": ifo, "nn_ref": Sws,
              "emb_sc": emb_sc, "km_centers": km.cluster_centers_,
              "profiles": profiles, "caselib": caselib, "thr": thr,
              "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    pickle.dump(bundle, open(os.path.join(MODELS, name + ".pkl"), "wb"))
    log(f"✅ 训练完成 → models/{name}.pkl (可到「测试」页使用)")


# ---------------- 测试 ----------------
_cache = {}
def load_bundle(name):
    if name not in _cache:
        b = pickle.load(open(os.path.join(MODELS, name + ".pkl"), "rb"))
        model = M.MEMRich(b["fields"])
        model.load_state_dict(b["state"])
        model.eval()
        b["model"] = model
        _cache[name] = b
    return _cache[name]


def score_account(bundle, rec):
    from sklearn.neighbors import NearestNeighbors
    b = bundle
    u = encode_rec(rec, b["gq"])
    pcs = M.per_position_ce(b["model"], [u], "cpu")
    zs = apply_znorm(pcs, b["zn"])
    zsum = sum(zs[k][0] for k in zs)
    c1m, c1t = float(zsum.mean()), M.topk_mean(zsum, 5)
    S = b["st_sc"].transform([stats_feats(u)])
    c2 = float(-b["ifo"].score_samples(S)[0])
    nn = NearestNeighbors(n_neighbors=5).fit(b["nn_ref"])
    c4 = float(nn.kneighbors(S)[0][0, -1])
    pcts = {"mem_mean": pct_among(c1m, b["c1m_w"]),
            "mem_top5": pct_among(c1t, b["c1t_w"]),
            "iforest": pct_among(c2, b["c2_w"]),
            "knn": pct_among(c4, b["c4_w"])}
    fused = float(np.mean([pcts["mem_mean"], pcts["iforest"], pcts["knn"]]))
    alarmed = fused > b["thr"]

    E = b["emb_sc"].transform(embed(b["model"], [u], "cpu"))
    d = np.linalg.norm(b["km_centers"] - E[0], axis=1)
    pk = int(d.argmin())
    pop = dict(b["profiles"][pk]); pop["dist"] = float(d[pk])
    pn = b.get("pop_names", {}).get(pk)
    if pn:
        pop["name"] = pn["name"]
        pop["definition"] = pn["definition"]

    fp = surprise_profile([u], zs)
    mo = None
    if alarmed and b.get("fp_km"):
        Xq = b["fp_km"]["scaler"].transform(fp)
        dk = np.linalg.norm(b["fp_km"]["centers"] - Xq[0], axis=1)
        mk = int(dk.argmin())
        info = b["fp_km"]["names"].get(mk, {"name": f"手法簇{mk}"})
        mo = {"k": mk, "name": info.get("name"),
              "definition": info.get("definition", ""),
              "dist": float(dk[mk])}

    attr = None
    if b["caselib"] is not None:
        Xq = b["caselib"]["scaler"].transform(fp)
        nn10 = NearestNeighbors(n_neighbors=11).fit(b["caselib"]["X"])
        dd, ii = nn10.kneighbors(Xq)
        qid = rec.get("user_id")
        nbs = [{"uid": b["caselib"]["uid"][j],
                "black": bool(b["caselib"]["is_black"][j]),
                "btype": b["caselib"]["btype"][j],
                "dist": float(dd[0][t])} for t, j in enumerate(ii[0])
               if b["caselib"]["uid"][j] != qid][:10]   # 排除查询账户自己
        bl = [n["btype"] for n in nbs if n["black"]]
        # 置信度: 与库内白样本互相之间的典型近邻距离比较
        if "d_ref" not in b:
            Xw = b["caselib"]["X"][~b["caselib"]["is_black"]][:300]
            nnw = NearestNeighbors(n_neighbors=2).fit(Xw)
            b["d_ref"] = float(np.median(nnw.kneighbors(Xw)[0][:, 1]))
        low_conf = bool(nbs and nbs[0]["dist"] > 3 * b["d_ref"])
        attr = {"neighbors": nbs,
                "black_frac": len(bl) / 10,
                "majority": max(set(bl), key=bl.count) if bl else None,
                "low_conf": low_conf, "d_ref": b["d_ref"]}

    top = np.argsort(-zsum)[:5]
    evs = rec["events"][:M.MAX_LEN]
    top_events = [{"i": int(i), "t": evs[i]["t"], "type": evs[i]["type"],
                   "amount": evs[i].get("amount"),
                   "channel": evs[i].get("channel", ""),
                   "result": evs[i].get("result", ""),
                   "z": float(zsum[i])} for i in sorted(top.tolist())]
    warning = None
    if len(evs) < 15:
        warning = (f"该账户只有 {len(evs)} 笔行为，低于可靠评分所需的最少 15 笔——"
                   "所有通道的分数统计意义不足，报警应视为『历史不足+初始行为可疑，转人工核验』，"
                   "而非确定性判断。")
    return {"uid": rec.get("user_id", "?"), "n_events": len(evs),
            "warning": warning, "mo": mo,
            "alarmed": bool(alarmed), "fused": fused, "thr": b["thr"],
            "channels": pcts, "population": pop, "attribution": attr,
            "top_events": top_events,
            "truth": {"label": rec.get("label"),
                      "wtype": rec.get("wtype"), "btype": rec.get("btype")}}


# ---------------- Web 服务 ----------------
TRAIN = {"running": False, "log": []}


def _train_thread(data, name, epochs):
    TRAIN["log"] = []
    TRAIN["running"] = True
    try:
        train_bundle(data, name, epochs,
                     lambda s: TRAIN["log"].append(s))
    except Exception as e:
        TRAIN["log"].append(f"❌ 训练失败: {e!r}")
    finally:
        TRAIN["running"] = False


def _name_thread(name):
    TRAIN["log"] = []
    TRAIN["running"] = True
    try:
        import name_clusters
        name_clusters.name_bundle(name, lambda s: TRAIN["log"].append(s))
        _cache.pop(name, None)          # 让下次评分读到新类别
    except Exception as e:
        TRAIN["log"].append(f"❌ 定类失败: {e!r}")
    finally:
        TRAIN["running"] = False


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(p.query)
        if p.path == "/":
            body = open(os.path.join(HERE, "app.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p.path == "/api/datasets":
            self._json(sorted(f for f in os.listdir(HERE)
                              if f.endswith(".jsonl")))
        elif p.path == "/api/models":
            out = []
            for f in sorted(os.listdir(MODELS)):
                if f.endswith(".pkl"):
                    try:
                        b = pickle.load(open(os.path.join(MODELS, f), "rb"))
                        out.append({"name": f[:-4], "data": b.get("data"),
                                    "trained_at": b.get("trained_at"),
                                    "has_caselib": b.get("caselib") is not None,
                                    "has_names": "pop_names" in b})
                    except Exception:
                        pass
            self._json(out)
        elif p.path == "/api/trainlog":
            self._json(TRAIN)
        elif p.path == "/api/accounts":
            data = q["data"][0]
            out = []
            for l in open(os.path.join(HERE, data), encoding="utf-8"):
                r = json.loads(l)
                out.append({"uid": r["user_id"], "label": r.get("label"),
                            "wtype": r.get("wtype"),
                            "btype": r.get("btype")})
            rng = random.Random(0)
            blacks = [a for a in out if a["label"] == 1]
            whites = [a for a in out if a["label"] == 0]
            rng.shuffle(whites)
            self._json(blacks + whites[:60])
        else:
            self._json({"err": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/train":
            if TRAIN["running"]:
                return self._json({"err": "已有训练在进行"}, 409)
            data = os.path.join(HERE, req["data"])
            threading.Thread(target=_train_thread,
                             args=(data, req["name"],
                                   int(req.get("epochs", 30))),
                             daemon=True).start()
            self._json({"ok": True})
        elif self.path == "/api/name":
            if TRAIN["running"]:
                return self._json({"err": "已有任务在进行"}, 409)
            threading.Thread(target=_name_thread, args=(req["model"],),
                             daemon=True).start()
            self._json({"ok": True})
        elif self.path == "/api/score":
            try:
                b = load_bundle(req["model"])
                if "record" in req:
                    rec = req["record"]
                else:
                    rec = None
                    for l in open(os.path.join(HERE, req["data"]),
                                  encoding="utf-8"):
                        r = json.loads(l)
                        if r["user_id"] == req["uid"]:
                            rec = r
                            break
                    if rec is None:
                        return self._json({"err": "账户不存在"}, 404)
                self._json(score_account(b, rec))
            except Exception as e:
                self._json({"err": repr(e)}, 500)
        else:
            self._json({"err": "not found"}, 404)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        data, name = sys.argv[2], sys.argv[3]
        ep = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        train_bundle(data, name, ep, lambda s: print(s, flush=True))
    else:
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        print(f"http://127.0.0.1:{port}", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
