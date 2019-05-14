#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ランダムフォレスト判別実行スクリプト
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

# データの読み込み（暫定的に iris のデータ）
data = load_iris()
X = data.data
y = data.target
label = data.feature_names

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

# ランダムフォレストによる判別
clf = RandomForestClassifier(n_estimators = 1000, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 予測性能を表示
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 記述子の寄与率上位 10 種を表示
important_rank = np.argsort(clf.feature_importances_)[::-1]
for i in important_rank:
    print(label[i])
    print(clf.feature_importances_[i])

