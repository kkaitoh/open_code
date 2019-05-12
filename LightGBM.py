#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Light GBMによる回帰モデル（GBDTR）実行スクリプト

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import optuna
from functools import partial

# データの読み込み（暫定的に Boston のデータ）
data = load_boston()
X = data.data
y = data.target
label = data.feature_names

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# optuna による hyperparametr の最適化関数を設定
def optimized_function(X, y, trial):
    
    # 探索するハイパーパラメーターの範囲を決める
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leavies": trial.suggest_int("num_leavies", 2, 100),
        "min_data_in_leaf": trial.suggest_int("num_leavies", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.1, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 7)
    }

    # クロスバリデーションの条件設定
    validation_result = cross_validate(lgb.LGBMRegressor(**params, random_state = 0), X, y, scoring = "r2", 
                                       cv = KFold(n_splits = 5,  random_state = 0), return_train_score = True)
    
    print(validation_result)
    
    # バリデーション内の予測指標 (r2_score) の平均値を返す。
    return - np.mean(validation_result["test_score"])

# セッションを作成
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))
f = partial(optimized_function, X_train, y_train)

# ベイズ最適化を実行
study.optimize(f, n_trials = 100)

# 最適なパラメーターを選択
best_params = study.best_params

print(best_params)

# モデル構築
rgr = lgb.LGBMRegressor(**best_params, random_state = 0)
rgr.fit(X_train, y_train)
y_pred = rgr.predict(X_test)

# 予測性能を表示
print(r2_score(y_test, y_pred))

# 記述子の寄与率上位 10 種を表示
important_rank = np.argsort(rgr.feature_importances_)[::-1]
for i in important_rank:
    print(label[i])

# y-y プロット
plt.figure(figsize = ([8, 8]))
plt.xlim([-10, 60])
plt.ylim([-10, 60])

plt.plot([-10, 60], [-10, 60], c = "k", linewidth = 0.7)
plt.scatter(y_test, y_pred, c = "dodgerblue", s = 20)

plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[ ]:




