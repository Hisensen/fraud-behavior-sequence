# -*- coding: utf-8 -*-
"""用用户自己的样本数据, 逐步打印流水线每个处理阶段的真实中间结果。"""
import json
from datetime import datetime

import numpy as np

EVENT_TYPES = ["登录", "查余额", "改限额", "改密码", "绑卡", "解绑卡",
               "设备变更", "转入", "转出", "消费", "还款", "借款"]
T2I = {e: i for i, e in enumerate(EVENT_TYPES)}
GAP_EDGES = [60, 300, 1800, 3600, 21600, 86400, 604800]
GAP_LBL = ["≤1m", "1-5m", "5-30m", "0.5-1h", "1-6h", "6-24h", "1-7d", ">7d", "首"]
RATIO_EDGES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

users = [json.loads(l) for l in open("user_sample.jsonl", encoding="utf-8")]

print("=" * 72)
print("STEP 0  原始数据格式检查")
print("=" * 72)
r = users[0]
print(f"记录数: {len(users)}  |  每条 = 一个用户: user_id + label + events 列表")
print(f"示例用户 {r['user_id']}: {len(r['events'])} 个事件")
print("首事件原文:", json.dumps(r["events"][0], ensure_ascii=False))
fields = set()
for u in users:
    for ev in u["events"]:
        fields |= set(ev.keys())
print("全部出现过的字段:", sorted(fields))
print("→ 有 type/t/amount/result; 无 channel/ip_change (统一Schema将填'不适用')")

print()
print("=" * 72)
print("STEP 1  全局统计量(正常应只在训练集白样本上算, 这里用4条样本演示)")
print("=" * 72)
all_amts = np.array([ev["amount"] for u in users for ev in u["events"]
                     if "amount" in ev])
gq = np.quantile(all_amts, np.linspace(0, 1, 8)[1:-1])
print(f"全部金额 {len(all_amts)} 笔, 范围 {all_amts.min():,.0f} ~ {all_amts.max():,.0f}")
print("全局金额 7 个分位切点(→8桶):")
print("  ", " | ".join(f"{q:,.0f}" for q in gq))

print()
print("=" * 72)
print(f"STEP 2  逐事件字段抽取与分桶 —— 用户 {r['user_id']} 全部 {len(r['events'])} 个事件")
print("=" * 72)
amts_u = [ev.get("amount") for ev in r["events"]]
nz = [a for a in amts_u if a]
med = np.median(nz)
print(f"该用户自己的金额中位数(个人基线) = {med:,.0f}\n")
hdr = f"{'#':>2} {'时间':<12} {'事件':<5} {'金额':>9} {'间隔':>7} |{'type':>5} {'res':>4} {'ch':>3} {'ip':>3} {'全局桶':>4} {'个人桶':>4} {'gap桶':>4} {'hour':>5}"
print(hdr)
print("-" * len(hdr))
prev = None
rows = []
for k, ev in enumerate(r["events"]):
    t = datetime.strptime(ev["t"], "%Y-%m-%d %H:%M:%S")
    gap_s = (t - prev).total_seconds() if prev else None
    gi = 8 if gap_s is None else int(np.searchsorted(GAP_EDGES, gap_s, side="left"))
    a = ev.get("amount")
    gamt = 0 if not a else 1 + int(np.searchsorted(gq, a))
    pamt = 0 if not a else 1 + int(np.searchsorted(RATIO_EDGES, a / med))
    res = {"成功": 1, "失败": 2}.get(ev.get("result"), 0)
    row = dict(ti=T2I[ev["type"]], res=res, ch=0, ip=0, gamt=gamt, pamt=pamt,
               gap=gi, hour=t.hour + t.minute / 60)
    rows.append(row)
    gap_disp = "—" if gap_s is None else (f"{gap_s/60:.0f}m" if gap_s < 3600
                                          else f"{gap_s/3600:.1f}h")
    print(f"{k:>2} {ev['t'][5:16]:<12} {ev['type']:<5} "
          f"{('—' if not a else format(a, ',')):>9} {gap_disp:>7} |"
          f"{row['ti']:>5} {res:>4} {0:>3} {0:>3} {gamt:>4} {pamt:>4} "
          f"{gi:>4} {row['hour']:>5.1f}")
    prev = t
print(f"\n桶含义: res 0=不适用/1=成功/2=失败;  ch/ip 全 0=字段缺失(统一Schema填法)")
print(f"gap桶: {dict(enumerate(GAP_LBL))}")
print(f"个人桶: 金额÷个人中位数({med:,.0f}), 切点 {RATIO_EDGES} → 1~7, 0=非金额事件")

print()
print("=" * 72)
print("STEP 3  查表拼接 → 模型输入向量 (以 #2 消费 167,313 为例)")
print("=" * 72)
ev2 = rows[2]
print(f"离散索引: type={ev2['ti']}(消费) res={ev2['res']} ch=0 ip=0 "
      f"gamt={ev2['gamt']} pamt={ev2['pamt']} gap={ev2['gap']}")
print(f"""
  type={ev2['ti']}  → 查 12×32 事件类型表  → 32 维向量
  res={ev2['res']}    → 查  3×4  结果表      →  4 维
  ch=0     → 查  4×8  渠道表      →  8 维 (0号=「不适用」的可学习向量)
  ip=0     → 查  3×4  ip表       →  4 维 (同上)
  gamt={ev2['gamt']}   → 查  8×8  全局金额表   →  8 维
  pamt={ev2['pamt']}   → 查  8×16 个人金额表   → 16 维
  gap={ev2['gap']}    → 查  9×16 间隔表      → 16 维
  hour={ev2['hour']:.1f} → sin/cos → Linear(2,8) →  8 维
  拼接 32+4+8+4+8+16+16+8 = 96 维 → Linear(96,64) → 该事件的 64 维输入向量""")
print(f"整条序列: {len(rows)} 个事件 × 64 维 → [L={len(rows)}, 64] 矩阵进 Transformer")
EOF_MARK = None
