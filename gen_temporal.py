# -*- coding: utf-8 -*-
"""
生成 temporal 数据集(v3): 黑白样本词频/金额/时段分布严格对齐,
唯一判别信号是"敏感操作的时序密集性"。

设计要点
--------
1. 事件类型数量: 黑白从同一分布采样(逐类型 Poisson), 保证词频对齐。
2. 金额: 逐类型同一 lognormal 分布, 黑白一致。
3. 小时分布: 所有事件(含黑样本爆发窗口)都从同一个白天分布采样起点。
4. 白样本: 普通事件按"会话"组织(会话内间隔 5s~10min, 制造大量小间隔),
   敏感事件彼此至少间隔 6 小时(孤立)。
5. 黑样本: 普通事件会话结构与白样本完全相同; 全部敏感事件 + 若干交易事件
   集中在一个 ~30 分钟的爆发窗口内, 按欺诈模式排序
   (盗号/套现/洗钱, sub_label = 1/2/3)。
6. 噪声: 10% 白样本出现一对 30~120 分钟内的敏感操作(半密集);
   10% 黑样本爆发窗口放宽到 ~2 小时(松散)。
"""
import json
import math
import random
from datetime import datetime, timedelta

random.seed(42)

EVENT_TYPES = ["登录", "查余额", "改限额", "改密码", "绑卡", "解绑卡",
               "设备变更", "转入", "转出", "消费", "还款", "借款"]
SENSITIVE = {"改限额", "改密码", "绑卡", "解绑卡", "设备变更"}
MONETARY = {"转入", "转出", "消费", "还款", "借款"}

T0 = datetime(2024, 1, 1)
WINDOW_DAYS = 30

# 逐类型数量分布 —— 黑白共用同一函数
def sample_counts(rng):
    return {
        "登录":   1 + poisson(rng, 18),
        "查余额": poisson(rng, 14),
        "消费":   2 + poisson(rng, 20),
        "转入":   1 + poisson(rng, 5),
        "转出":   1 + poisson(rng, 5),
        "还款":   poisson(rng, 2),
        "借款":   poisson(rng, 1.2),
        "设备变更": 1 + poisson(rng, 0.7),
        "改密码": 1 + poisson(rng, 0.7),
        "改限额": poisson(rng, 0.9),
        "绑卡":   poisson(rng, 0.9),
        "解绑卡": poisson(rng, 0.6),
    }

def poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1

# 金额分布 —— 逐类型同一 lognormal, 黑白一致
AMOUNT_PARAMS = {"转入": (7.5, 1.2), "转出": (7.5, 1.2), "消费": (5.5, 1.3),
                 "还款": (7.0, 0.8), "借款": (8.5, 0.9)}

def sample_amount(rng, etype):
    mu, sigma = AMOUNT_PARAMS[etype]
    return round(min(math.exp(rng.gauss(mu, sigma)), 500000), 2)

def daytime_hour(rng):
    """所有事件共用的白天小时分布 N(14,4), 截断到 [7,23]"""
    return min(23.0, max(7.0, rng.gauss(14, 4)))

def rand_time(rng):
    day = rng.uniform(0, WINDOW_DAYS - 1)
    return T0 + timedelta(days=day, hours=daytime_hour(rng),
                          minutes=rng.uniform(0, 59))

def make_event(rng, etype, t):
    ev = {"type": etype, "t": t.strftime("%Y-%m-%d %H:%M:%S")}
    if etype in MONETARY:
        ev["amount"] = sample_amount(rng, etype)
        ev["channel"] = rng.choice(["APP", "WEB", "POS"])
    else:
        ev["result"] = "成功" if rng.random() > 0.03 else "失败"
    return ev

def place_sessions(rng, ordinary):
    """把普通事件切成若干会话, 会话内间隔 5s~10min —— 黑白共用"""
    rng.shuffle(ordinary)
    events = []
    i = 0
    while i < len(ordinary):
        size = min(rng.randint(2, 6), len(ordinary) - i)
        t = rand_time(rng)
        for j in range(size):
            events.append(make_event(rng, ordinary[i + j], t))
            t += timedelta(seconds=min(600, max(5, rng.expovariate(1 / 90))))
        i += size
    return events

def gen_white(rng, uid, noisy):
    counts = sample_counts(rng)
    ordinary = [e for e, c in counts.items() if e not in SENSITIVE for _ in range(c)]
    events = place_sessions(rng, ordinary)

    # 敏感事件: 彼此至少间隔 6 小时
    sens = [e for e, c in counts.items() if e in SENSITIVE for _ in range(c)]
    placed = []
    for etype in sens:
        for _ in range(200):
            t = rand_time(rng)
            if all(abs((t - p).total_seconds()) >= 6 * 3600 for p in placed):
                placed.append(t)
                events.append(make_event(rng, etype, t))
                break
    # 噪声: 半密集敏感对(30~120 分钟)
    if noisy and len(placed) >= 1:
        base = rng.choice(placed)
        t = base + timedelta(minutes=rng.uniform(30, 120))
        events.append(make_event(rng, rng.choice(sorted(SENSITIVE)), t))

    events.sort(key=lambda e: e["t"])
    return {"user_id": uid, "label": 0, "sub_label": 0, "events": events}

FRAUD_PATTERNS = {
    1: ["设备变更", "改密码", "改限额", "转出", "转出", "解绑卡"],          # 盗号
    2: ["绑卡", "借款", "消费", "消费", "消费", "转出", "解绑卡"],          # 套现
    3: ["转入", "转入", "转入", "改限额", "转出", "转出", "解绑卡"],        # 洗钱
}

def gen_black(rng, uid, sub_label, noisy):
    counts = sample_counts(rng)
    budget = {e: c for e, c in counts.items()}

    # 按模式从预算里取事件组成爆发序列(预算不够的类型跳过, 保证词频对齐)
    burst_types = []
    for etype in FRAUD_PATTERNS[sub_label]:
        if budget.get(etype, 0) > 0:
            burst_types.append(etype)
            budget[etype] -= 1
    # 剩余敏感事件也全部并入爆发(黑样本敏感操作全部密集)
    for etype in sorted(SENSITIVE):
        while budget.get(etype, 0) > 0:
            burst_types.append(etype)
            budget[etype] -= 1

    # 爆发窗口: 正常 ~30 分钟内; 噪声样本放宽到 ~2 小时
    gap_mean = 900 if noisy else 120
    gap_cap = 3600 if noisy else 300
    t = rand_time(rng)
    burst_events = []
    for etype in burst_types:
        burst_events.append(make_event(rng, etype, t))
        t += timedelta(seconds=min(gap_cap, max(20, rng.expovariate(1 / gap_mean))))

    ordinary = [e for e, c in budget.items() if e not in SENSITIVE for _ in range(c)]
    events = place_sessions(rng, ordinary) + burst_events
    events.sort(key=lambda e: e["t"])
    return {"user_id": uid, "label": 1, "sub_label": sub_label, "events": events}

def main():
    rng = random.Random(42)
    records = []
    for i in range(500):
        records.append(gen_white(rng, f"W_{i:04d}", noisy=rng.random() < 0.10))
    for i in range(500):
        records.append(gen_black(rng, f"B_{i:04d}", sub_label=1 + i % 3,
                                 noisy=rng.random() < 0.10))
    rng.shuffle(records)
    with open("data_temporal.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 对齐性自检 ----
    from collections import Counter
    def stats(recs):
        cnt, gaps, hours = Counter(), [], []
        for r in recs:
            prev = None
            for ev in r["events"]:
                cnt[ev["type"]] += 1
                t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
                hours.append(t.hour)
                if prev is not None:
                    gaps.append((t - prev).total_seconds())
                prev = t
        n = len(recs)
        return {e: cnt[e] / n for e in EVENT_TYPES}, gaps, hours

    w = [r for r in records if r["label"] == 0]
    b = [r for r in records if r["label"] == 1]
    fw, gw, hw = stats(w)
    fb, gb, hb = stats(b)
    print(f"{'事件类型':<8}{'白均值':>10}{'黑均值':>10}{'相对差':>10}")
    for e in EVENT_TYPES:
        rel = abs(fw[e] - fb[e]) / max(fw[e], 1e-9)
        print(f"{e:<8}{fw[e]:>10.2f}{fb[e]:>10.2f}{rel:>9.1%}")
    def gap_hist(gs):
        edges = [60, 300, 1800, 3600, 21600, 86400, 604800]
        buckets = [0] * (len(edges) + 1)
        for g in gs:
            k = sum(g > e for e in edges)
            buckets[k] += 1
        tot = sum(buckets)
        return [x / tot for x in buckets]
    print("\n间隔分桶占比(≤1m,1-5m,5-30m,0.5-1h,1-6h,6-24h,1-7d,>7d):")
    print("  白:", " ".join(f"{x:.3f}" for x in gap_hist(gw)))
    print("  黑:", " ".join(f"{x:.3f}" for x in gap_hist(gb)))
    print(f"\n平均小时: 白 {sum(hw)/len(hw):.2f}  黑 {sum(hb)/len(hb):.2f}")
    print(f"平均序列长度: 白 {sum(len(r['events']) for r in w)/len(w):.1f}  "
          f"黑 {sum(len(r['events']) for r in b)/len(b):.1f}")
    print(f"\n共 {len(records)} 条 → data_temporal.jsonl")

if __name__ == "__main__":
    main()
