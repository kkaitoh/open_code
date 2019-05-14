#!/usr/bin/env python
# coding: utf-8

# In[17]:


# 線形判別実行スクリプト
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# オートスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 線形判別
clf = LinearDiscriminantAnalysis()
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# 予測性能を表示
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 判別式の傾きと切片を表示
print(clf.coef_)
print(clf.intercept_)

