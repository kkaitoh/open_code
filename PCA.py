#!/usr/bin/env python
# coding: utf-8

# PCA 実行スクリプト

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

# データの読み込み（暫定的に iris のデータ）
data = load_iris()
X = data.data
y = data.target

# オートスケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# プロット
x_0 = X_pca[:, 0]
x_1 = X_pca[:, 1]

plt.figure(figsize = ([8, 8]))
plt.xlim([-5, 5])
plt.ylim([-5, 5])

# 目盛の感覚を指定
plt.xticks(np.arange(-5, 6, 1))
plt.yticks(np.arange(-5, 6, 1))

# 各ラベルごとに色分け
plt.scatter(x_0[y == 0], x_1[y == 0], c = "dodgerblue", s = 20)
plt.scatter(x_0[y == 1], x_1[y == 1], c = "crimson", s = 20)
plt.scatter(x_0[y == 2], x_1[y == 2], c = "limegreen", s = 20)

#  フォントの調整
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

