# -*- coding: utf-8 -*-
"""思想12·Behavior DNA(潜变量): VAE 学统计特征的潜在行为因子。
检测=重构误差; 聚类=潜变量μ; 可解释性=各因子与已知属性的相关。
Baseline: PCA 重构误差 / 原始统计特征聚类。"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import bp_common as C

torch.manual_seed(42); np.random.seed(42)
d = C.load_art()
tr = d["is_train"]
print("== 思想12 Behavior DNA(VAE) ==", flush=True)
sc = StandardScaler().fit(d["stats"][tr])
X = torch.tensor(sc.transform(d["stats"]), dtype=torch.float32)
Xtr = X[torch.tensor(tr)]
D, Z = X.shape[1], 8


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, Z * 2))
        self.dec = nn.Sequential(nn.Linear(Z, 64), nn.ReLU(), nn.Linear(64, D))

    def forward(self, x):
        mu, lv = self.enc(x).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.dec(z), mu, lv


m = VAE()
opt = torch.optim.Adam(m.parameters(), 1e-3)
for ep in range(300):
    idx = torch.randperm(len(Xtr))
    for s in range(0, len(Xtr), 128):
        xb = Xtr[idx[s:s + 128]]
        xr, mu, lv = m(xb)
        loss = ((xr - xb) ** 2).sum(1).mean() + \
            0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(1).mean() * 0.1
        opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    xr, mu, _ = m(X)
    rec = ((xr - X) ** 2).mean(1).numpy()
    MU = mu.numpy()
C.row("VAE 重构误差检测", C.det_auc(d, rec))
C.row("VAE 潜变量聚类", None, C.white_ari(d, MU), C.black_ari(d, MU))

p = PCA(8).fit(sc.transform(d["stats"][tr]))
Xs = sc.transform(d["stats"])
rec_p = ((Xs - p.inverse_transform(p.transform(Xs))) ** 2).mean(1)
C.row("PCA-8 重构误差 baseline", C.det_auc(d, rec_p))
C.row("原始统计特征聚类 baseline", None, C.white_ari(d, d["stats"]),
      C.black_ari(d, d["stats"]))

# 因子可解释性: 每个潜因子与哪个统计维相关最强
names = ([f"词频:{t}" for t in
          ["登录", "查余额", "改限额", "改密码", "绑卡", "解绑卡",
           "设备变更", "转入", "转出", "消费", "还款", "借款"]] +
         [f"间隔桶{i}" for i in range(10)] + [f"个金桶{i}" for i in range(9)] +
         ["APP", "WEB", "POS", "凌晨占比", "时刻sin", "时刻cos",
          "失败率", "换IP率", "log事件数"])
Xn = sc.transform(d["stats"])
for z in range(Z):
    r = [abs(np.corrcoef(MU[:, z], Xn[:, j])[0, 1]) for j in range(D)]
    j = int(np.argmax(r))
    print(f"    因子{z}: 最相关={names[j]} (|r|={r[j]:.2f})", flush=True)
C.save_score("e12_vae", rec)
