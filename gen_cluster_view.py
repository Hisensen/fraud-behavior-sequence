# -*- coding: utf-8 -*-
"""生成 data_cluster 数据设计检视页 docs/cluster-data-view.html:
每种人群/手法一个真实样本(攻击行标红) + 设计参数 + 分组实测统计。"""
import json
from datetime import datetime

import numpy as np

recs = [json.loads(l) for l in open("data_cluster.jsonl", encoding="utf-8")]

DESIGN = {
    "工薪族": "中等金额基线(e^6.5≈665) · APP · 白天13:30±3h · 消费20/登录14/还款2.5 为主",
    "学生党": "低基线(e^4.8≈122) · APP · 晚间21:00±2h · 高频小额消费28/借款1.0",
    "个体商户": "较高基线(e^7.0≈1097) · POS · 营业时间12:00±4h · 转入16/查余额16/转出7",
    "理财大户": "高基线(e^8.3≈4024) · WEB · 上午10:00±2.5h · 查余额20/低频大额转入转出3.5",
    "盗号爆发": "随机时刻: 2-3次登录失败(换IP)→成功→改密/设备变更×2→2-4笔转出(8~18×个人基线), 全程20分钟内, 非偏好渠道",
    "慢速抽干": "受害者自己操作: 第8-13天起, 每0.8-1.8天一组[登录→1-5分钟后转出2~5×基线], 共12-16组, 无失败无换IP, 偏好渠道",
    "养卡套现": "全月分布 8-12 组[转入X → 5-60分钟后 转出/消费≈0.9X], X=1~3×基线, WEB/POS 混用",
    "赌博出款": "5-8个凌晨活跃夜(0-4点): 每夜3-6笔小额转入(0.5~2×基线,间隔2-20分钟) + 70%概率跟一笔大额转出(4~8×基线)",
}


def stats(rs, black=False):
    ne = [len(r["events"]) for r in rs]
    night, fail, ip = [], [], []
    natk, mult = [], []
    for r in rs:
        evs = r["events"]
        hrs = [int(e["t"][11:13]) for e in evs]
        night.append(sum(1 for h in hrs if h < 6) / len(evs))
        fail.append(sum(1 for e in evs if e.get("result") == "失败"))
        ip.append(sum(1 for e in evs if e.get("ip_change") == 1))
        if black:
            atk = [e for e in evs if e.get("atk")]
            natk.append(len(atk))
            base = np.median([e["amount"] for e in evs
                              if "amount" in e and not e.get("atk")] or [1])
            m = [e["amount"] / base for e in atk if "amount" in e]
            if m:
                mult.append(np.median(m))
    row = (f"<td>{len(rs)}</td><td>{np.mean(ne):.0f}</td>"
           f"<td>{np.mean(night):.1%}</td><td>{np.mean(fail):.2f}</td>"
           f"<td>{np.mean(ip):.2f}</td>")
    if black:
        row += f"<td>{np.mean(natk):.1f}</td><td>{np.median(mult):.1f}×</td>"
    else:
        row += "<td>-</td><td>-</td>"
    return row


def sample_table(r):
    evs = r["events"]
    base = np.median([e["amount"] for e in evs
                      if "amount" in e and not e.get("atk")] or [1])
    rows = []
    prev = None
    for i, e in enumerate(evs):
        t = datetime.strptime(e["t"], "%Y-%m-%d %H:%M:%S")
        gap = "-" if prev is None else (
            f"{(t-prev).total_seconds()/60:.0f}分" if (t-prev).total_seconds() < 5400
            else f"{(t-prev).total_seconds()/3600:.0f}时" if (t-prev).total_seconds() < 172800
            else f"{(t-prev).days}天")
        prev = t
        amt = f"{e['amount']:.0f}" if "amount" in e else ""
        mul = f"{e['amount']/base:.1f}×" if "amount" in e else ""
        cls = ' class="atk"' if e.get("atk") else ""
        night = ' 🌙' if t.hour < 6 else ''
        rows.append(
            f"<tr{cls}><td>{i}</td><td>{e['t'][5:16]}{night}</td><td>{gap}</td>"
            f"<td>{e['type']}</td><td>{amt}</td><td>{mul}</td>"
            f"<td>{e.get('channel','')}</td><td>{e.get('result','')}</td>"
            f"<td>{'⚡' if e.get('ip_change')==1 else ''}</td></tr>")
    return (f"<p class='sub'>个人金额中位数(非攻击) ≈ <b>{base:.0f}</b> 元 · "
            f"共 {len(evs)} 事件, 攻击事件 "
            f"{sum(1 for e in evs if e.get('atk'))} 个(红行)</p>"
            "<div class='tbl'><table><tr><th>#</th><th>时间</th><th>距上笔</th>"
            "<th>类型</th><th>金额</th><th>×个人中位</th><th>渠道</th>"
            "<th>结果</th><th>IP</th></tr>" + "".join(rows) + "</table></div>")


# 选样本: 每子类取事件数最接近该类均值的一个
def pick(rs):
    m = np.mean([len(r["events"]) for r in rs])
    return min(rs, key=lambda r: abs(len(r["events"]) - m))


parts = []
parts.append("""<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>data_cluster 数据设计检视</title><style>
:root{--bg:#faf9f6;--card:#fff;--ink:#1e2528;--sub:#5b6b70;--line:#e3ded4;
--teal:#00997C;--teal-bg:#e6f5f1;--red:#C24A32;--red-bg:#faece8;--code:#f0ede6}
@media (prefers-color-scheme:dark){:root{--bg:#14181a;--card:#1d2326;--ink:#e8e6e0;
--sub:#9aa8ad;--line:#323b40;--teal:#2fbf9e;--teal-bg:#12312a;--red:#e0745c;
--red-bg:#3a201a;--code:#262d31}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.8 -apple-system,"PingFang SC",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:30px 20px 80px}
h1,h2{font-family:"Songti SC",serif}h1{font-size:30px;margin:6px 0}
h2{font-size:22px;border-bottom:2px solid var(--teal);padding-bottom:6px;margin:44px 0 8px}
.sub{color:var(--sub);font-size:13.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px 20px;margin:16px 0}
.tbl{max-height:430px;overflow:auto;border:1px solid var(--line);border-radius:10px;margin:8px 0}
table{border-collapse:collapse;font-size:12.5px;width:100%}
th,td{border-bottom:1px solid var(--line);padding:3px 9px;text-align:left;white-space:nowrap}
th{background:var(--code);position:sticky;top:0}
tr.atk td{background:var(--red-bg);color:var(--red);font-weight:600}
.design{background:var(--teal-bg);border-radius:8px;padding:6px 12px;font-size:13.5px;margin:6px 0}
nav{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);
overflow-x:auto;white-space:nowrap;z-index:5;margin:0 -20px 8px;}
nav a{color:var(--sub);text-decoration:none;font-size:13px;padding:4px 10px;border-radius:99px}
nav a:hover{background:var(--teal-bg);color:var(--teal)}
</style><div class="wrap">
<div style="color:var(--teal);font-weight:700;letter-spacing:.15em;font-size:12.5px">DATA_CLUSTER · 设计检视</div>
<h1>聚类基准数据：设计是否合理，自己看</h1>
<p class="sub">2000 白(4人群×500) + 240 黑(4手法×60) · 每子类展示一个"最典型"样本(事件数最接近该类均值) ·
红行=注入的攻击事件 · 🌙=凌晨0-6点 · ⚡=换IP · 生成器 gen_cluster_data.py(种子固定可复现)</p>
<nav>""")
for k in DESIGN:
    parts.append(f"<a href='#{k}'>{k}</a>")
parts.append("<a href='#stats'>分组统计</a></nav>")

parts.append("<h2 id='stats'>分组实测统计（先看全貌）</h2><div class='card'><div class='tbl'>"
             "<table><tr><th>子类</th><th>人数</th><th>平均事件</th><th>凌晨事件占比</th>"
             "<th>人均失败次数</th><th>人均换IP</th><th>人均攻击事件</th><th>攻击金额中位倍数</th></tr>")
for wt in ["工薪族", "学生党", "个体商户", "理财大户"]:
    rs = [r for r in recs if r["wtype"] == wt and r["label"] == 0]
    parts.append(f"<tr><td>白·{wt}</td>{stats(rs)}</tr>")
for bt in ["盗号爆发", "慢速抽干", "养卡套现", "赌博出款"]:
    rs = [r for r in recs if r.get("btype") == bt]
    parts.append(f"<tr><td>黑·{bt}</td>{stats(rs, black=True)}</tr>")
parts.append("</table></div><p class='sub'>检查点: 赌博出款凌晨占比应显著高; 盗号失败/换IP应高; "
             "慢速抽干应与白样本几乎无异(只有攻击事件数和金额倍数暴露它); 黑白平均事件数应接近(词频部分对齐)。</p></div>")

for k, spec in DESIGN.items():
    is_black = k in ("盗号爆发", "慢速抽干", "养卡套现", "赌博出款")
    if is_black:
        rs = [r for r in recs if r.get("btype") == k]
    else:
        rs = [r for r in recs if r["wtype"] == k and r["label"] == 0]
    r = pick(rs)
    tag = "黑" if is_black else "白"
    base_note = f" · 底座人群: {r['wtype']}" if is_black else ""
    parts.append(f"<h2 id='{k}'>{tag} · {k} <span class='sub'>样本 {r['user_id']}{base_note}</span></h2>")
    parts.append(f"<div class='design'><b>设计</b>: {spec}</div>")
    parts.append("<div class='card'>" + sample_table(r) + "</div>")

parts.append("<p class='sub'>配套: blueprint-matrix.html(实验矩阵) · BLUEPRINT_MATRIX.md · "
             "gen_cluster_data.py 顶部 docstring 有完整设计说明</p></div>")

open("docs/cluster-data-view.html", "w", encoding="utf-8").write("".join(parts))
print("docs/cluster-data-view.html 生成完毕")
