#!/usr/bin/env python
# coding: utf-8

# 線形重回帰（MLR）実行スクリプト

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

# データの読み込み（暫定的に Boston のデータ）
data = load_boston()
X = data.data
y = data.target

X_train = X[np.array(list(range(0, X.shape[0]))) % 10 != 0, :]
X_test = X[np.array(list(range(0, X.shape[0]))) % 10 == 0, :]
y_train = y[np.array(list(range(0, X.shape[0]))) % 10 != 0]
y_test = y[np.array(list(range(0, X.shape[0]))) % 10 == 0]

# オートスケーリング
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train[:, np.newaxis])
y_test_scaled = scaler_y.transform(y_test[:, np.newaxis])

# モデル構築
rgr = LinearRegression()
rgr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = rgr.predict(X_test_scaled)

# リスケーリング
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# r2 score を表示
print(r2_score(y_test, y_pred))

# 構築したモデルの係数を表示
print(rgr.coef_)
print(rgr.intercept_)

# y-y プロットを作成
fig = plt.figure(figsize = ([8, 8]))
ax = fig.add_subplot(1,1,1)
plt.xlim([-10, 60])
plt.ylim([-10, 60])

# 目盛の感覚を指定
plt.xticks(np.arange(-10, 60, 10))
plt.yticks(np.arange(-10, 60, 10))

#  フォントの調整
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20

# データをプロット
plt.plot([-10, 60], [-10, 60], c = "k", linewidth = 0.7)
plt.scatter(y_test, y_pred, c = "dodgerblue", s = 20)

plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

