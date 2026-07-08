# -*- coding: utf-8 -*-
"""
生成带"子类真值"的数据 data_cluster.jsonl —— 验证聚类可行性专用。

白样本 4 种人群原型(各 500), 记录 wtype:
  工薪族: 中等基线, APP, 白天, 消费/还款为主
  学生党: 低基线, APP, 晚间, 高频小额消费, 偶尔借款
  个体商户: 高频转入+POS消费, POS 渠道, 营业时间
  理财大户: 高基线, WEB, 低频大额转入转出, 高频查余额

黑样本 4 种作案手法(各 60), 每个叠加在随机人群原型底座上, 记录 btype+wtype:
  盗号爆发: 登录失败→异地成功→改密/设备变更→20分钟内大额转出(8-18×s)
  慢速抽干: 受害者被诱导, 自己操作, 半月内 12-16 笔 2-5×s 转出, 无失败无换IP
  养卡套现: 8-12 组"转入X→几十分钟内转出/消费≈X"快进快出对
  赌博出款: 凌晨 0-5 点高频小额转入 + 间歇大额转出(4-8×s)

四种手法故意设计成不同的"惊讶指纹": 盗号→res+gap 头, 慢速抽干→pamt 头,
养卡→gap+type 头, 赌博→type+gap(凌晨条件结构)。
"""
import json
import math
import random
from datetime import datetime, timedelta

random.seed(7)

CHANNELS = ["APP", "WEB", "POS"]
T0 = datetime(2024, 1, 1)
DAYS = 30

ARCH = {
    "工薪族": dict(s_mu=6.5, s_sd=0.4, pref="APP", hour=13.5, hour_sd=3.0,
                   counts={"登录": 14, "查余额": 10, "消费": 20, "转入": 4,
                           "转出": 4, "还款": 2.5, "借款": 0.3, "设备变更": 0.5,
                           "改密码": 0.5, "改限额": 0.3, "绑卡": 0.3, "解绑卡": 0.2}),
    "学生党": dict(s_mu=4.8, s_sd=0.4, pref="APP", hour=21.0, hour_sd=2.0,
                   counts={"登录": 18, "查余额": 14, "消费": 28, "转入": 2,
                           "转出": 1.5, "还款": 0.3, "借款": 1.0, "设备变更": 0.6,
                           "改密码": 0.5, "改限额": 0.2, "绑卡": 0.4, "解绑卡": 0.3}),
    "个体商户": dict(s_mu=7.0, s_sd=0.4, pref="POS", hour=12.0, hour_sd=4.0,
                     counts={"登录": 10, "查余额": 16, "消费": 8, "转入": 16,
                             "转出": 7, "还款": 1.5, "借款": 0.8, "设备变更": 0.4,
                             "改密码": 0.4, "改限额": 0.5, "绑卡": 0.3, "解绑卡": 0.2}),
    "理财大户": dict(s_mu=8.3, s_sd=0.4, pref="WEB", hour=10.0, hour_sd=2.5,
                     counts={"登录": 8, "查余额": 20, "消费": 5, "转入": 3.5,
                             "转出": 3.5, "还款": 0.5, "借款": 0.2, "设备变更": 0.4,
                             "改密码": 0.5, "改限额": 0.4, "绑卡": 0.3, "解绑卡": 0.2}),
}
MONETARY = {"转入", "转出", "消费", "还款", "借款"}
AMT_MULT = {"消费": (-0.6, 0.5), "转出": (0.0, 0.5), "转入": (0.0, 0.5),
            "还款": (-0.2, 0.4), "借款": (-0.2, 0.4)}


def poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def base_events(rng, a):
    """按人群原型生成一个月的正常事件(未排序的 (dt, dict) 列表)。"""
    s = math.exp(rng.gauss(a["s_mu"], a["s_sd"]))
    evs = []
    for et, lam in a["counts"].items():
        for _ in range(poisson(rng, lam)):
            day = rng.random() * DAYS
            hour = rng.gauss(a["hour"], a["hour_sd"]) % 24
            dt = T0 + timedelta(days=day, hours=hour,
                                minutes=rng.random() * 60)
            ev = {"type": et, "channel":
                  a["pref"] if rng.random() < 0.85 else rng.choice(CHANNELS)}
            if et in MONETARY:
                mu, sd = AMT_MULT[et]
                ev["amount"] = round(s * math.exp(rng.gauss(mu, sd)), 2)
            if et == "登录":
                ev["result"] = "成功" if rng.random() < 0.97 else "失败"
                ev["ip_change"] = 1 if rng.random() < 0.05 else 0
            else:
                ev["result"] = "成功"
            evs.append((dt, ev))
    return s, evs


def mk_ev(dt, et, ch, amount=None, result="成功", ip=None):
    ev = {"type": et, "channel": ch, "result": result}
    if amount is not None:
        ev["amount"] = round(amount, 2)
    if ip is not None:
        ev["ip_change"] = ip
    elif et == "登录":
        ev["ip_change"] = 0
    return dt, ev


def attack_盗号(rng, s, pref):
    ch = rng.choice([c for c in CHANNELS if c != pref])
    t = T0 + timedelta(days=5 + rng.random() * 20, hours=rng.uniform(0, 24))
    evs = []
    for _ in range(rng.randint(2, 3)):
        evs.append(mk_ev(t, "登录", ch, result="失败", ip=1))
        t += timedelta(seconds=rng.uniform(20, 90))
    evs.append(mk_ev(t, "登录", ch, result="成功", ip=1))
    t += timedelta(seconds=rng.uniform(30, 120))
    for et in rng.sample(["设备变更", "改密码", "改限额"], 2):
        evs.append(mk_ev(t, et, ch))
        t += timedelta(seconds=rng.uniform(30, 180))
    for _ in range(rng.randint(2, 4)):
        evs.append(mk_ev(t, "转出", ch, amount=s * rng.uniform(8, 18)))
        t += timedelta(seconds=rng.uniform(60, 300))
    return evs


def attack_慢速抽干(rng, s, pref):
    evs = []
    day = 8 + rng.random() * 5
    for _ in range(rng.randint(12, 16)):
        t = T0 + timedelta(days=day, hours=rng.gauss(15, 3) % 24)
        evs.append(mk_ev(t, "登录", pref, ip=0))
        evs.append(mk_ev(t + timedelta(minutes=rng.uniform(1, 5)),
                         "转出", pref, amount=s * rng.uniform(2, 5)))
        day += rng.uniform(0.8, 1.8)
        if day > DAYS - 1:
            break
    return evs


def attack_养卡套现(rng, s, pref):
    evs = []
    for _ in range(rng.randint(8, 12)):
        day = rng.random() * DAYS
        t = T0 + timedelta(days=day, hours=rng.gauss(14, 4) % 24)
        x = s * rng.uniform(1.0, 3.0)
        evs.append(mk_ev(t, "转入", rng.choice(["WEB", "POS"]), amount=x))
        t2 = t + timedelta(minutes=rng.uniform(5, 60))
        out_et = rng.choice(["转出", "消费"])
        evs.append(mk_ev(t2, out_et, rng.choice(["WEB", "POS"]),
                         amount=x * rng.uniform(0.9, 0.99)))
    return evs


def attack_赌博出款(rng, s, pref):
    evs = []
    for _ in range(rng.randint(5, 8)):          # 若干个凌晨活跃夜
        day = rng.random() * DAYS
        t = T0 + timedelta(days=int(day), hours=rng.uniform(0, 4))
        for _ in range(rng.randint(3, 6)):
            evs.append(mk_ev(t, "转入", pref, amount=s * rng.uniform(0.5, 2)))
            t += timedelta(minutes=rng.uniform(2, 20))
        if rng.random() < 0.7:
            evs.append(mk_ev(t + timedelta(minutes=rng.uniform(5, 30)),
                             "转出", pref, amount=s * rng.uniform(4, 8)))
    return evs


ATTACKS = {"盗号爆发": attack_盗号, "慢速抽干": attack_慢速抽干,
           "养卡套现": attack_养卡套现, "赌博出款": attack_赌博出款}


def finalize(evs):
    evs.sort(key=lambda x: x[0])
    out = []
    for dt, ev in evs:
        e = {"type": ev["type"], "t": dt.strftime("%Y-%m-%d %H:%M:%S")}
        e.update({k: v for k, v in ev.items() if k != "type"})
        out.append(e)
    return out


def main():
    rng = random.Random(7)
    recs = []
    uid = 0
    for wtype, a in ARCH.items():
        for _ in range(500):
            _, evs = base_events(rng, a)
            recs.append({"user_id": f"W{uid:05d}", "label": 0,
                         "wtype": wtype, "btype": None,
                         "events": finalize(evs)})
            uid += 1
    for btype, fn in ATTACKS.items():
        for _ in range(60):
            wtype = rng.choice(list(ARCH))
            a = ARCH[wtype]
            s, evs = base_events(rng, a)
            atk = fn(rng, s, a["pref"])
            for _, ev in atk:
                ev["atk"] = 1          # 标记攻击事件(仅供检视, 模型不读)
            # 词频部分对齐: 攻击消耗基础预算
            n_drop = min(len(atk) // 2, len(evs) - 10)
            if n_drop > 0:
                idx = set(rng.sample(range(len(evs)), n_drop))
                evs = [e for i, e in enumerate(evs) if i not in idx]
            recs.append({"user_id": f"B{uid:05d}", "label": 1,
                         "wtype": wtype, "btype": btype,
                         "events": finalize(evs + atk)})
            uid += 1
    rng.shuffle(recs)
    with open("data_cluster.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ev = sum(len(r["events"]) for r in recs)
    print(f"写出 {len(recs)} 用户 / {n_ev} 事件 -> data_cluster.jsonl")
    for wt in ARCH:
        ls = [len(r["events"]) for r in recs if r["wtype"] == wt and r["label"] == 0]
        print(f"  白·{wt:<6} n={len(ls)}  平均事件 {sum(ls)/len(ls):.0f}")
    for bt in ATTACKS:
        ls = [len(r["events"]) for r in recs if r.get("btype") == bt]
        print(f"  黑·{bt:<6} n={len(ls)}  平均事件 {sum(ls)/len(ls):.0f}")


if __name__ == "__main__":
    main()
