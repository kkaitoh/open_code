#!/usr/bin/env python
# coding: utf-8

# LASSO実行スクリプト

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

import optuna
import matplotlib.pyplot as plt
from functools import partial

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

# optuna による hyperparametr の最適化関数を設定
def optimized_function(X, y, trial):
    
    # 探索するハイパーパラメーターの範囲を決める
    params = {
        "alpha": trial.suggest_uniform("alpha", 0.01, 1)
    }

    # クロスバリデーションの条件設定
    validation_result = cross_validate(Lasso(**params), X, y, scoring = "r2", 
                                       cv = KFold(n_splits = 5,  random_state = 0),return_train_score = True)
    
    # バリデーション内の予測指標 (r2_score) の平均値を返す。
    print( - np.mean(validation_result["test_score"]))

    return - np.mean(validation_result["test_score"])

# セッションを作成
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))
f = partial(optimized_function, X_train_scaled, y_train)

# ベイズ最適化を実行
study.optimize(f, n_trials = 30)

# 最適なパラメーターを選択
best_params = study.best_params

# モデル構築
rgr = Lasso(**best_params, random_state = 0)
rgr.fit(X_train_scaled, y_train)
y_pred = rgr.predict(X_test_scaled)

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

