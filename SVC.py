#!/usr/bin/env python
# coding: utf-8

# In[25]:


# サポートベクターマシンによる判別モデル（SVC）実行スクリプト

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import optuna

from functools import partial


# In[26]:


# データの読み込み（暫定的に iris のデータ）
data = load_iris()
X = data.data
y = data.target
label = data.feature_names


# In[27]:


# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# In[28]:


# データをオートスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[49]:


# optuna による hyperparametr の最適化関数を設定
def optimized_function(X, y, trial):
    
    # 探索するハイパーパラメーターの範囲を決める
    params = {
        "C": trial.suggest_loguniform("C", 2e-7, 2e7) ,
        "gamma": trial.suggest_loguniform("gamma", 2e-7, 2e-1)
    }
    
    # クロスバリデーションの条件設定
    validation_result = cross_validate(SVC(**params ,random_state = 0), X, y, scoring = "accuracy",
                                       cv = StratifiedKFold(n_splits = 5,  random_state = 0), return_train_score = True)
    
    # バリデーション内の予測指標 (r2_score) の平均値を返す。
    return - np.mean(validation_result["test_score"])


# In[50]:


# セッションを作成
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))
f = partial(optimized_function, X_train, y_train)

# ベイズ最適化を実行
study.optimize(f, n_trials = 10)

# 最適なパラメーターを選択
best_params = study.best_params


# In[53]:


# 最適化されたパラメーターと全トレーニングデータを用いてモデル構築
clf = SVC(**best_params)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)


# In[55]:


# 予測性能を表示
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


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




