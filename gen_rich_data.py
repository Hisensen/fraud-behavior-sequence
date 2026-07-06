# -*- coding: utf-8 -*-
"""
生成"富字段"基准数据 data_rich.jsonl —— 对齐用户方案的统一 Schema。

每个白样本用户有"个人画像": 金额基线 s_u(跨用户差几个数量级)、偏好渠道、
活跃时段。事件字段: type / t / amount / channel / result / ip_change。

黑样本 = 盗号(账户接管), 攻击信号分层埋进四类字段:
  ① result 层: 攻击开始连续 2-3 次登录失败后成功
  ② channel/ip 层: 攻击期渠道切到该用户的非偏好渠道, 登录 ip_change=1
  ③ 个人金额层: 攻击转出金额 = 8~18 × 该用户基线 s_u ——
     对低基线用户是天文数字, 但落在全局金额分布内部(富人正常转账也这么大),
     全局分桶难以区分, 只有"相对个人基线"能看见
  ④ 时序层: 攻击事件在 ~20 分钟窗口内密集发生
噪声: 10% 白样本有一笔 6-10× 个人基线的"反常大额消费"(单点巧合);
      10% 黑样本是"隐蔽型"(倍数 3-6, 不换渠道, 只失败 1 次)。
词频对齐: 攻击消耗用户自身的事件预算(登录/转出/敏感操作从预算扣除)。
"""
import json
import math
import random
from datetime import datetime, timedelta

random.seed(2024)

CHANNELS = ["APP", "WEB", "POS"]
SENSITIVE = {"改限额", "改密码", "绑卡", "解绑卡", "设备变更"}
MONETARY = {"转入", "转出", "消费", "还款", "借款"}
T0 = datetime(2024, 1, 1)
DAYS = 30


def poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def sample_counts(rng):
    return {"登录": 3 + poisson(rng, 14), "查余额": poisson(rng, 12),
            "消费": 2 + poisson(rng, 18), "转入": 1 + poisson(rng, 4),
            "转出": 2 + poisson(rng, 4), "还款": poisson(rng, 2),
            "借款": poisson(rng, 1),
            "设备变更": 1 + poisson(rng, 0.5), "改密码": 1 + poisson(rng, 0.6),
            "改限额": poisson(rng, 0.4), "绑卡": poisson(rng, 0.4),
            "解绑卡": poisson(rng, 0.3)}


AMT_MULT = {"消费": (0.0, 0.6), "转出": (1.0, 0.6), "转入": (1.0, 0.6),
            "还款": (0.8, 0.5), "借款": (0.8, 0.5)}


class Profile:
    def __init__(self, rng):
        self.s = math.exp(rng.gauss(6.2, 1.0))          # 个人金额基线
        self.pref = rng.choice(CHANNELS)                 # 偏好渠道
        self.hour_mu = rng.gauss(13.5, 2.5)              # 活跃时段


def p_hour(rng, prof):
    return min(23.0, max(7.0, rng.gauss(prof.hour_mu, 2.5)))


def rand_time(rng, prof, day_lo=0, day_hi=DAYS - 1):
    return T0 + timedelta(days=rng.uniform(day_lo, day_hi),
                          hours=p_hour(rng, prof), minutes=rng.uniform(0, 59))


def amount_of(rng, prof, etype, mult=None):
    mu, sig = AMT_MULT[etype]
    a = prof.s * math.exp(rng.gauss(mu, sig)) if mult is None else prof.s * mult
    return round(min(a, 800000), 2)


def channel_of(rng, prof, attack=False):
    if attack:
        others = [c for c in CHANNELS if c != prof.pref]
        return rng.choice(others) if rng.random() < 0.85 else prof.pref
    return prof.pref if rng.random() < 0.75 else rng.choice(CHANNELS)


def make_event(rng, prof, etype, t, attack=False, amt_mult=None,
               force_result=None, ip=None):
    ev = {"type": etype, "t": t.strftime("%Y-%m-%d %H:%M:%S")}
    if etype in MONETARY:
        ev["amount"] = amount_of(rng, prof, etype, amt_mult)
        ev["channel"] = channel_of(rng, prof, attack)
    else:
        ev["result"] = force_result or ("成功" if rng.random() > 0.03 else "失败")
    if etype == "登录":
        ev["ip_change"] = ip if ip is not None else (1 if rng.random() < 0.06 else 0)
    return ev


def sessions(rng, prof, ordinary):
    rng.shuffle(ordinary)
    evs, i = [], 0
    while i < len(ordinary):
        size = min(rng.randint(2, 6), len(ordinary) - i)
        t = rand_time(rng, prof)
        for j in range(size):
            evs.append(make_event(rng, prof, ordinary[i + j], t))
            t += timedelta(seconds=min(900, max(20, rng.expovariate(1 / 150))))
        i += size
    return evs


def place_sensitive(rng, prof, sens, evs):
    placed = []
    for etype in sens:
        for _ in range(200):
            t = rand_time(rng, prof)
            if all(abs((t - p).total_seconds()) >= 6 * 3600 for p in placed):
                placed.append(t)
                evs.append(make_event(rng, prof, etype, t))
                break


def gen_white(rng, uid):
    prof = Profile(rng)
    counts = sample_counts(rng)
    resetup = rng.random() < 0.25   # 换手机重置: 合法的密集敏感操作爆发
    if resetup:
        for k, v in {"登录": 1, "设备变更": 1, "改密码": 1, "绑卡": 1}.items():
            counts[k] = max(0, counts[k] - v)

    ordinary = [e for e, c in counts.items() if e not in SENSITIVE
                for _ in range(c)]
    evs = sessions(rng, prof, ordinary)
    sens = [e for e, c in counts.items() if e in SENSITIVE for _ in range(c)]
    place_sensitive(rng, prof, sens, evs)

    if resetup:
        # 合法爆发: 一次登录成功(新设备ip变化) → 设备变更 → 改密码 → 绑卡
        # 与盗号的区别: 无连续失败登录、渠道仍是偏好渠道、之后没有大额转出
        t = rand_time(rng, prof)
        g = lambda lo, hi: timedelta(seconds=rng.uniform(lo, hi))
        evs.append(make_event(rng, prof, "登录", t, force_result="成功", ip=1))
        t += g(40, 150)
        evs.append(make_event(rng, prof, "设备变更", t, force_result="成功"))
        t += g(60, 240)
        evs.append(make_event(rng, prof, "改密码", t, force_result="成功"))
        t += g(60, 300)
        evs.append(make_event(rng, prof, "绑卡", t, force_result="成功"))

    if rng.random() < 0.10:   # 噪声: 单笔反常大额消费
        t = rand_time(rng, prof)
        evs.append(make_event(rng, prof, "消费", t,
                              amt_mult=rng.uniform(6, 10)))
    evs.sort(key=lambda e: e["t"])
    return {"user_id": uid, "label": 0, "events": evs}


def gen_black(rng, uid):
    prof = Profile(rng)
    counts = sample_counts(rng)
    subtle = rng.random() < 0.10
    # 攻击消耗预算, 保持词频对齐
    n_fail = 1 if subtle else rng.randint(2, 3)
    budget_use = {"登录": n_fail + 1, "转出": 2, "设备变更": 1, "改密码": 1,
                  "解绑卡": 1 if counts["解绑卡"] > 0 else 0}
    for k, v in budget_use.items():
        counts[k] = max(0, counts[k] - v)

    ordinary = [e for e, c in counts.items() if e not in SENSITIVE
                for _ in range(c)]
    evs = sessions(rng, prof, ordinary)
    sens = [e for e, c in counts.items() if e in SENSITIVE for _ in range(c)]
    place_sensitive(rng, prof, sens, evs)

    # 攻击窗口
    t = T0 + timedelta(days=rng.uniform(19, 28), hours=p_hour(rng, prof),
                       minutes=rng.uniform(0, 59))
    gap = lambda lo, hi: timedelta(seconds=rng.uniform(lo, hi))
    for _ in range(n_fail):
        evs.append(make_event(rng, prof, "登录", t, attack=True,
                              force_result="失败", ip=1))
        t += gap(30, 90)
    evs.append(make_event(rng, prof, "登录", t, attack=True,
                          force_result="成功", ip=1))
    t += gap(40, 120)
    evs.append(make_event(rng, prof, "设备变更", t, attack=True,
                          force_result="成功"))
    t += gap(60, 200)
    evs.append(make_event(rng, prof, "改密码", t, attack=True,
                          force_result="成功"))
    t += gap(60, 240)
    mlo, mhi = (3, 6) if subtle else (8, 18)
    for _ in range(2):
        evs.append(make_event(rng, prof, "转出", t, attack=not subtle,
                              amt_mult=rng.uniform(mlo, mhi)))
        t += gap(60, 300)
    if budget_use["解绑卡"]:
        evs.append(make_event(rng, prof, "解绑卡", t, attack=True,
                              force_result="成功"))
    evs.sort(key=lambda e: e["t"])
    return {"user_id": uid, "label": 1, "events": evs}


def main():
    rng = random.Random(2024)
    recs = ([gen_white(rng, f"W_{i:04d}") for i in range(1500)] +
            [gen_black(rng, f"B_{i:04d}") for i in range(500)])
    rng.shuffle(recs)
    with open("data_rich.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 对齐性自检 ----
    from collections import Counter
    import numpy as np
    def stats(rs):
        cnt, amts_out, hours = Counter(), [], []
        for r in rs:
            for ev in r["events"]:
                cnt[ev["type"]] += 1
                if ev["type"] == "转出":
                    amts_out.append(ev["amount"])
                hours.append(int(ev["t"][11:13]))
        return cnt, np.array(amts_out), hours
    w = [r for r in recs if r["label"] == 0]
    b = [r for r in recs if r["label"] == 1]
    cw, aw, hw = stats(w); cb, ab, hb = stats(b)
    print(f"{'类型':<8}{'白均':>8}{'黑均':>8}")
    for e in sorted(cw, key=lambda x: -cw[x]):
        print(f"{e:<8}{cw[e]/len(w):>8.2f}{cb[e]/len(b):>8.2f}")
    print(f"\n转出金额(全局): 白 p50={np.median(aw):,.0f} p90={np.percentile(aw,90):,.0f}"
          f" | 黑 p50={np.median(ab):,.0f} p90={np.percentile(ab,90):,.0f}")
    print(f"平均小时: 白 {sum(hw)/len(hw):.1f} 黑 {sum(hb)/len(hb):.1f}")
    print(f"序列长度: 白 {sum(len(r['events']) for r in w)/len(w):.1f} "
          f"黑 {sum(len(r['events']) for r in b)/len(b):.1f}")
    print(f"\n共 {len(recs)} 条 → data_rich.jsonl")


if __name__ == "__main__":
    main()
