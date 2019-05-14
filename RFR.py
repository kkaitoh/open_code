#!/usr/bin/env python
# coding: utf-8

# In[7]:


# ランダムフォレストによる回帰モデル（RFR）実行スクリプト

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

# データの読み込み（暫定的に Boston のデータ）
data = load_boston()
X = data.data
y = data.target
label = data.feature_names

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# モデル構築
rgr = RandomForestRegressor(n_estimators = 1000, random_state = 0)
rgr.fit(X_train, y_train)
y_pred = rgr.predict(X_test)

# 予測性能を表示
print(r2_score(y_test, y_pred))

# 記述子の寄与率上位 10 種を表示
important_rank = np.argsort(rgr.feature_importances_)[::-1]
for i in important_rank:
    print(label[i])
    print(rgr.feature_importances_[i])

# y-y プロット
plt.figure(figsize = ([8, 8]))
plt.xlim([-10, 60])
plt.ylim([-10, 60])

#  フォントの調整
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20

plt.plot([-10, 60], [-10, 60], c = "k", linewidth = 0.7)
plt.scatter(y_test, y_pred, c = "dodgerblue", s = 20)

plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

