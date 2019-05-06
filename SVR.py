#!/usr/bin/env python
# coding: utf-8

# In[153]:


# サポートベクターマシンによる回帰モデル（SVR）実行スクリプト

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import optuna

from functools import partial


# In[154]:


# データの読み込み（暫定的に Boston のデータ）
data = load_boston()
X = data.data
y = data.target
label = data.feature_names


# In[155]:


# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[156]:


# トレーニングデータをさらにモデル構築用データとバリデーションデータに分割する
X_model, X_validation, y_model, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)


# In[165]:


# データをオートスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# y_scaler = StandardScaler()
# y_train_scaled = y_scaler.fit_transform(y_train)
# y_test_scaled = y_scaler.transform(y_test)


# In[211]:


# optuna による hyperparametr の最適化関数を設定
def optimized_function(X, y, trial):
    
    # 探索するハイパーパラメーターの範囲を決める
    params = {
        "C": trial.suggest_loguniform("C", 2e-7, 2e7) ,
        "epsilon": trial.suggest_loguniform("epsilon", 2e-7, 2e-1),
        "gamma": trial.suggest_loguniform("gamma", 2e-7, 2e-1)
    }

    # クロスバリデーションの条件設定
    validation_result = cross_validate(SVR(**params), X, y, scoring = "r2", 
                                       cv = KFold(n_splits = 5,  random_state = 0),return_train_score = True)
    
    # バリデーション内の予測指標 (r2_score) の平均値を返す。
    print( - np.mean(validation_result["test_score"]))
    return - np.mean(validation_result["test_score"])


# In[212]:


# セッションを作成
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))
f = partial(optimized_function, X_train_scaled, y_train)

# ベイズ最適化を実行
study.optimize(f, n_trials = 50)

# 最適なパラメーターを選択
best_params = study.best_params


# In[213]:


# 最適化されたパラメーターと全トレーニングデータを用いてモデル構築
rgr = SVR(**best_params)
rgr.fit(X_train_scaled, y_train)
y_pred = rgr.predict(X_test_scaled)


# In[214]:


# 予測性能を表示
print(r2_score(y_test, y_pred))


# In[205]:


# 記述子の寄与率上位 10 種を表示
important_rank = np.argsort(rgr.feature_importances_)[::-1]
for i in important_rank:
    print(label[i])


# In[162]:


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




