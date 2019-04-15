#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.contrib import rnn
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from functools import partial


# In[2]:


# データセットの読み込み
X_train_df = pd.read_csv("/Users/kaitoh/Documents/Dataset/train_scaled_mordred_descriptor.csv")
X_train = np.array(X_train_df)
descriptor_label = X_train_df.columns

y_train = np.array(pd.read_csv("/Users/kaitoh/Documents/Dataset/train_AR_active.csv"))
X_test = np.array(pd.read_csv("/Users/kaitoh/Documents/Dataset/test_scaled_mordred_descriptor.csv"))
y_test = np.array(pd.read_csv("/Users/kaitoh/Documents/Dataset/test_AR_active.csv"))


# In[10]:


y_train = y_train[:, 1]
y_test = y_test[:, 1]


# In[14]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 1000, random_state = 0, class_weight = "balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[15]:


print(matthews_corrcoef(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))


# In[3]:


X_steric_train = np.load("/Users/kaitoh/Documents/Dataset/tox21_train_atom_steric_2.npy")
X_steric_test = np.load("/Users/kaitoh/Documents/Dataset/tox21_test_atom_steric_2.npy")


# In[ ]:


X_steric_model = 
y_model = 
X_steric_validation = 
y_validation = 


# In[50]:


y_train.tolist()


# In[84]:


X_steric_train = X_steric_train.astype("float32")
X_steric_test = X_steric_test.astype("float32")
y_train_categorical = to_categorical(y_train.tolist())
y_test_categorical = to_categorical(y_test.tolist())


# In[90]:


# RNN を試す
def rnn_prediction(X_train, y_train, X_test, y_test, node_num = 10, learning_rate = 0.1, batch_size = 32, epochs = 3, dropout = 0.1):
    
    model = keras.Sequential()
    model.add(layers.LSTM(node_num, batch_input_shape = (None, X_train.shape[1], X_train.shape[2]), dropout = dropout))
    model.add(layers.Dense(2))
    model.add(layers.Activation(tf.nn.softmax))
    
    optimizer = tf.keras.optimizers.RMSprop(lr = learning_rate, decay = 1e-6)
    model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = )
    
    model.fit(X_train,  y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
    y_pred = model.predict(X_test)
    return matthews_corrcoef(y_test, y_pred)


# In[80]:


X_steric_test[3,4,4]


# In[91]:


# ベイズ最適化によるハイパーパラメーターの決定
# optuna による hyperparametr の最適化
def optimized_function(X, y, X_val, y_val, trial):
    params = {
        "learning_rate": trial.suggest_uniform("learning_rate", 0.001, 0.500) ,
        "node_num": trial.suggest_int("node_num", 2, 32),
        "dropout": trial.suggest_uniform("dropout", 0.01, 0.3)
    }
    
    y_pred = rnn_prediction(X, y, X_val, y_val, **params)
    return - matthews_corrcoef(y_val, y_pred)

# セッションを作成
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))

f = partial(optimized_function, X_steric_train, y_train_categorical, X_steric_test, y_test_categorical)
study.optimize(f, n_trials = 3)


# In[60]:


study.best_params

