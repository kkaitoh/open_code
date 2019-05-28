#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 決定木判別実行スクリプト
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
from dtreeviz.trees import dtreeviz


# In[2]:


# データの読み込み（暫定的に iris のデータ）
data = load_iris()
X = data.data
y = data.target
label = data.feature_names

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# In[3]:


# 決定木による判別
clf = DecisionTreeClassifier(random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 予測性能を表示
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[4]:


# 記述子の寄与率上位 10 種を表示
important_rank = np.argsort(clf.feature_importances_)[::-1]
for i in important_rank:
    print(label[i])
    print(clf.feature_importances_[i])

