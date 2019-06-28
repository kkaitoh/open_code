#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, Dropout, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score


# In[60]:


#  画像データを読み込み Cifer10 を今回は用いる
train_data, test_data = cifar10.load_data()


# In[92]:


# トレーニングデータとテストデータを用意
train_X = train_data[0]
train_y = train_data[1]

test_X = test_data[0]
test_y = test_data[1]

# トレーニングデータの各クラスを one hot vector で表記
train_y = to_categorical(train_y)

# 簡便化のための前処理。深い意味はない。
test_y = test_y.T[0]


# In[99]:


# データセットの大きさを確認
train_X.shape


# In[62]:


# 入力データの次元（縦と横と色（RGB））を指定
input_dimension = (train_X.shape[1], train_X.shape[2], train_X.shape[3])


# In[90]:


# CNN モデルの形状を構築 データの形状は VGG16 をイメージ
# Sequential のインスタンスを用意し、そこに順番に入れてくやり方がシンプル
model = Sequential()
model.add(InputLayer(input_dimension, name = "Input"))

# 畳み込みは filter の数と大きさを指定する
model.add(Conv2D(8, (3, 3), name = "Conv1"))

#  各層に dropout と batchnormalization を指定
model.add(Dropout(0.1, name = "Dropout1"))
model.add(BatchNormalization(name = "Normalize1"))
model.add(Conv2D(8, (3, 3), name = "Conv2"))
model.add(Dropout(0.1, name = "Dropout2"))
model.add(BatchNormalization(name = "Normalize2"))

# プーリング層は大きさを指定、最大値以外を用いたい場合は別の関数を参照
model.add(MaxPool2D((2, 2), name = "Pool1"))

# Padding  を行なってみる
model.add(Conv2D(16, (3, 3), padding = "same", name = "Conv3"))
model.add(Dropout(0.1, name = "Dropout3"))
model.add(BatchNormalization(name = "Normalize3"))
model.add(Conv2D(16, (3, 3), padding = "same", name = "Conv4"))
model.add(Dropout(0.1, name = "Dropout4"))
model.add(BatchNormalization(name = "Normalize4"))
model.add(MaxPool2D((2, 2), name = "Pool2"))

# 最後に全結合モデルを構築してクラス分類を行う
# 事前にデータを二次元に変換する
model.add(Flatten(name = "Flatten"))
model.add(Dense(10, activation = "sigmoid", name = "Dense"))

# モデルと設定（学習方法）をコンパイルすることで使えるようになる
optimizer = Adam(lr = 0.001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy")


# In[91]:


# モデルの形状を確認
model.summary()


# In[94]:


# モデルを学習させる。エポック数とバッチサイズはここで指定
model.fit(train_X, train_y, epochs = 3, batch_size = 128)


# In[95]:


# 予測を行う。ただの predict だと one hot vector として返ってくるので、 predict_classes を用いる
pred_y = model.predict_classes(test_X)


# In[96]:


# テストデータの accuracy を算出
accuracy_score(test_y, pred_y)

