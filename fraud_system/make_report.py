# -*- coding: utf-8 -*-
"""读 system_result.json → 渲染 report.html: 每个方法在本数据上的实测效果。"""
import json

R = json.load(open("system_result.json", encoding="utf-8"))
f = lambda v: f"{v:.4f}"
p = lambda v: f"{v:.1%}"

def bar(v, best=1.0):
    w = max(2, v / best * 100)
    color = "var(--teal)" if v >= 0.9 else ("var(--gold)" if v >= 0.75 else "var(--red)")
    return (f"<div class='bar'><div style='width:{w:.0f}%;background:{color}'></div>"
            f"<span>{f(v)}</span></div>")

s = R["setup"]; m7 = R["m7"]
singles = m7["singles"]

html = ["""<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>一体化系统实测报告 · data_cluster</title><style>
:root{--bg:#faf9f6;--card:#fff;--ink:#1e2528;--sub:#5b6b70;--line:#e3ded4;
--teal:#00997C;--teal-bg:#e6f5f1;--red:#C24A32;--red-bg:#faece8;
--gold:#9a7b2d;--gold-bg:#f7f1df;--blue:#2d5f9a;--blue-bg:#e8eff8;--code:#f0ede6}
@media (prefers-color-scheme:dark){:root{--bg:#14181a;--card:#1d2326;--ink:#e8e6e0;
--sub:#9aa8ad;--line:#323b40;--teal:#2fbf9e;--teal-bg:#12312a;--red:#e0745c;
--red-bg:#3a201a;--gold:#cfa84e;--gold-bg:#332b16;--blue:#6fa3d8;--blue-bg:#1a2836;--code:#262d31}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15.5px/1.85 -apple-system,"PingFang SC",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:30px 20px 80px}
h1,h2{font-family:"Songti SC",serif}h1{font-size:30px;margin:6px 0}
h2{font-size:22px;border-bottom:2px solid var(--teal);padding-bottom:6px;margin:46px 0 8px}
h2 .no{display:inline-block;background:var(--teal);color:#fff;border-radius:8px;padding:0 10px;margin-right:8px;font-size:17px}
.sub{color:var(--sub);font-size:13.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px 22px;margin:14px 0}
table{border-collapse:collapse;font-size:13.5px;width:100%}
th,td{border:1px solid var(--line);padding:6px 11px;text-align:left}
th{background:var(--code);white-space:nowrap}
td.num{font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.twrap{overflow-x:auto;margin:10px 0}
.bar{position:relative;background:var(--code);border-radius:6px;height:22px;min-width:180px}
.bar div{height:100%;border-radius:6px}
.bar span{position:absolute;right:8px;top:0;font:12px/22px ui-monospace,Menlo,monospace}
.res{border-left:4px solid var(--teal);background:var(--teal-bg);border-radius:0 12px 12px 0;padding:10px 16px;margin:12px 0;font-size:14.5px}
.warn{border-left:4px solid var(--gold);background:var(--gold-bg);border-radius:0 12px 12px 0;padding:10px 16px;margin:12px 0;font-size:14.5px}
.big{font-size:26px;font-weight:800;color:var(--teal);font-family:ui-monospace,Menlo,monospace}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;text-align:center}
.kpi .sub{font-size:12.5px}
</style><div class="wrap">
<div style="color:var(--teal);font-weight:700;letter-spacing:.15em;font-size:12.5px">FRAUD_SYSTEM · 一体化实测</div>
<h1>7 个有效方法在我们数据上的完整实测</h1>"""]

html.append(f"<p class='sub'>数据 data_cluster.jsonl：{s['n_users']} 账户"
            f"（训练白 {s['n_train_w']} / 测试白 {s['n_test_w']} + 黑 {s['n_black']}）· "
            f"一条命令跑通全流水线（run_system.py）· 方法原理见 effective-methods.html</p>")

# 总览 KPI
al = R["alarm"]
html.append(f"""<div class="grid">
<div class="kpi"><div class="big">{f(m7['three'][0])}</div><div class="sub">三通道融合 AUC</div></div>
<div class="kpi"><div class="big">{al['n_black_hit']}/{al['black_total']}</div><div class="sub">报警池抓到的黑样本<br>(误报仅 {al['fp']} 个白)</div></div>
<div class="kpi"><div class="big">{f(R['m5']['ari'])}</div><div class="sub">人群聚类 ARI</div></div>
<div class="kpi"><div class="big">{p(R['m4b']['attr_hit'])}</div><div class="sub">案件库手法归因命中</div></div>
</div>""")

# 检测通道对比
html.append("<h2><span class='no'>A</span>检测：四个通道 + 融合，谁把黑样本排到了前面</h2><div class='card'><div class='twrap'><table>")
html.append("<tr><th>通道</th><th>AUC（可视化）</th><th>抓获率@误报1%</th></tr>")
rows = [("① MEM 惊讶度 (mean 池化)", R["m1"]["mean"]),
        ("① MEM 惊讶度 (top-5 池化)", R["m1"]["top5"]),
        ("② 统计特征 + iForest", R["m2"]["auc"]),
        ("③ 原型距离 K=16", R["m3"]["k16"]),
        ("④a 第5近白样本距离", R["m4a"]["auc"]),
        ("⑦ 三通道秩融合 (①mean+②+④a)", m7["three"]),
        ("⑦ 全通道秩融合", m7["all"])]
for name, (a, r1) in rows:
    hl = " style='background:var(--teal-bg)'" if name.startswith("⑦") else ""
    html.append(f"<tr{hl}><td>{name}</td><td>{bar(a)}</td><td class='num'>{p(r1)}</td></tr>")
html.append("</table></div>")
html.append(f"<p class='sub'>单中心对照：原型 K=1 的 AUC 只有 {f(R['m3']['k1_auc'])}——"
            f"再次证明\"全体客户一个中心\"必然失败。</p></div>")
html.append(f"""<div class="res"><b>报警池实况</b>：融合分超过白样本 99 分位的账户共 {al['n_alarm']} 个，
其中 {al['n_black_hit']} 个是真黑样本（全部 {al['black_total']} 个黑样本抓到 {p(al['n_black_hit']/al['black_total'])}），
只误报 {al['fp']} 个正常账户。这就是"误报率控制在 1%"的实际含义。</div>""")

# 方法5 人群
html.append("<h2><span class='no'>B</span>人群层：⑤ 嵌入聚类自动分出的四种人</h2><div class='card'>")
html.append(f"<p>2000 个白样本自动聚成 4 群，与真实人群吻合度 ARI = <b>{f(R['m5']['ari'])}</b>。每群的画像（机器自动算出，名字按多数真值标注）：</p><div class='twrap'><table>")
html.append("<tr><th>簇</th><th>人数</th><th>多数人群(纯度)</th><th>主渠道</th><th>凌晨占比</th></tr>")
for c in R["m5"]["clusters"]:
    html.append(f"<tr><td>簇{c['k']}</td><td class='num'>{c['n']}</td>"
                f"<td>{c['majority']} ({p(c['purity'])})</td>"
                f"<td>{c['top_channel']}</td><td class='num'>{p(c['night'])}</td></tr>")
html.append("</table></div><p class='sub'>落地用法：每群单独训练/单独定阈值（分群建模），新客户先归群再按本群标准审。</p></div>")

# 方法6 指纹
m6 = R["m6"]
html.append("<h2><span class='no'>C</span>归因层：⑥ 报警池黑样本的手法聚类</h2><div class='card'>")
html.append(f"<p>报警池中的 {m6['n_alarm_black']} 个黑样本，用惊讶画像聚成 4 簇，与真实手法吻合 ARI = <b>{f(m6['ari'])}</b>。混淆矩阵（行=真实手法，列=机器分的簇）：</p><div class='twrap'><table>")
html.append("<tr><th>真实手法</th><th>簇0</th><th>簇1</th><th>簇2</th><th>簇3</th></tr>")
for bt, row in m6["confusion"].items():
    cells = "".join(f"<td class='num'>{v if v else ''}</td>" for v in row)
    html.append(f"<tr><td>{bt}</td>{cells}</tr>")
html.append("</table></div></div>")

m4b = R["m4b"]
html.append(f"""<div class="res"><b>④b 案件库检索（{m4b['lib_size']} 个已结案入库）</b>：
新案子查 10 个最近邻 → 邻居里有案件的比例 Recall@10 = <b>{p(m4b['recall10'])}</b>，
按邻居多数手法归因命中 <b>{p(m4b['attr_hit'])}</b>，
邻居黑占比当风险分 AUC = <b>{f(m4b['auc'])}</b>。攒案件越多越准，且每个判断附带相似历史案件作证据。</div>""")

# 示例
html.append("<h2><span class='no'>D</span>逐账户示例：同一个账户在各通道眼中的排名</h2><div class='card'>")
html.append("<p>每种手法取融合分最高的账户 + 一个正常账户对照。数字 = 该通道把它排进测试集最可疑的前百分之几（越接近 100 越可疑）：</p><div class='twrap'><table>")
html.append("<tr><th>账户</th><th>真实身份</th><th>底座人群</th><th>①MEM</th><th>②iForest</th><th>④a近邻</th><th>是否报警</th></tr>")
for e in R["examples"]:
    q = lambda v: f"前{100-v:.0f}%" if v > 50 else f"后{v:.0f}%"
    mark = "🔴 报警" if e["alarmed"] else "🟢 放行"
    html.append(f"<tr><td class='num'>{e['uid']}</td><td>{e['btype']}</td><td>{e['wtype']}</td>"
                f"<td class='num'>{q(e['pct_m1'])}</td><td class='num'>{q(e['pct_m2'])}</td>"
                f"<td class='num'>{q(e['pct_m4'])}</td><td>{mark}</td></tr>")
html.append("</table></div><p class='sub'>读法：\"前3%\"= 该通道认为它比 97% 的账户可疑。不同手法在不同通道上暴露程度不同——这就是要融合的原因。</p></div>")

html.append("""<div class="warn"><b>口径说明</b>：本数据为带双真值的仿真基准（4 人群×4 手法），
手法会明显改变聚合统计，对 ②④ 类方法偏友好；MEM 的不可替代性在词频对齐数据与真实数据上验证（见实验矩阵页发现1）。
真实数据上线时四通道全保留，融合兜底。</div>
<p class="sub" style="margin-top:40px;border-top:1px solid var(--line);padding-top:14px">
fraud_system/ · 复现: python run_system.py && python make_report.py ·
方法详解: ../docs/effective-methods.html · 实验矩阵: ../docs/blueprint-matrix.html</p>
</div>""")

open("report.html", "w", encoding="utf-8").write("".join(html))
print("report.html 生成完毕")
