#!/usr/bin/env python
# coding: utf-8

# In[354]:


# Kernel PCA 実行スクリプト

import numpy as np
import pandas as pd

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_swiss_roll

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import optuna
from scipy.spatial import distance
from functools import partial


# In[261]:


# データの読み込み（暫定的に iris のデータ）
# data = load_iris()
X = data.data
# y = data.target

X, y = make_swiss_roll(1000, 10, random_state = 0)

# オートスケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[262]:


# k3n error を算出する関数。多次元空間におけるデータの近さと次元削減によりプロットされたデータの距離関係は同じという仮定に基づいた指標。
def k3n_error(X_multi, X_plane, k = 3):
    
    # 距離行列を算出
    X_multi_distance = distance.cdist(X_multi, X_multi)
    X_plane_distance = distance.cdist(X_plane, X_plane)

    # あるサンプルから近いサンプルの順番を求める
    X_multi_distance_rank = np.argsort(X_multi_distance, axis=1)
    X_plane_distance_rank = np.argsort(X_plane_distance, axis=1)

    # 0 を別の最小値に置き換える操作（スケーリングの操作でエラーを回避するため）
    for i in range(X_plane_distance.shape[0]):
        X_plane_distance[i, :][X_plane_distance[i, :] == 0] = np.min(X_plane_distance[i, X_plane_distance[i, :] != 0])
        X_multi_distance[i, :][X_multi_distance[i, :] == 0] = np.min(X_multi_distance[i, X_multi_distance[i, :] != 0])

    # 単位行列（選択に用いるだけ）
    I = np.eye(len(X_multi_distance), dtype=bool)

    # 多次元、二次元空間における近傍 k 個の点を変換した際の距離を選ぶ
    X_multi_nearest_to_plane = np.sort(X_plane_distance[:, X_multi_distance_rank[:, 1:3+1]][I])
    X_plane_nearest_to_multi = np.sort(X_multi_distance[:, X_plane_distance_rank[:, 1:3+1]][I])

    # 多次元、二次元平面における近傍 k 個の距離を抽出
    X_plane_nearest = np.sort(X_plane_distance)[:, 1:k+1]
    X_multi_nearest = np.sort(X_multi_distance)[:, 1:k+1]

    # 距離の差分を算出し、二次元における距離でスケーリング的処理をする
    k3nerror_to_plane = (X_multi_nearest_to_plane - X_plane_nearest) / X_plane_nearest
    k3nerror_to_multi = (X_plane_nearest_to_multi - X_multi_nearest) / X_multi_nearest

    # 全ての差分の総和を算出
    sum_k3nerror_to_plane = np.sum(k3nerror_to_plane)
    sum_k3nerror_to_multi = np.sum(k3nerror_to_multi)

    # データの数で割り、さらに近傍点の数 k で割ることで k3nerror を算出
    k3nerror_to_plane = sum_k3nerror_to_plane / k3nerror_to_plane.shape[0]/ k
    k3nerror_to_multi = sum_k3nerror_to_multi / k3nerror_to_multi.shape[0]/ k
    k3nerror = k3nerror_to_plane + k3nerror_to_multi
    return k3nerror


# In[293]:


# ベイズ最適化による Kernel PCA の最適化

# optuna による hyperparametr の最適化関数を設定
def optimized_function(X, trial):
    
    # 探索するハイパーパラメーターの範囲を決める
    params = {
        "gamma": trial.suggest_loguniform("gamma", 2**-10, 2**10)
    }
    
    # kernel PCA を実行
    kpca = KernelPCA(kernel = "rbf", **params, random_state = 0)
    X_kpca = kpca.fit_transform(X)
    X_kpca_plane = X_kpca[:,:2]

    # k3n error が小さくなるように探索する
    return k3n_error(X, X_kpca_plane)

# セッションを作成
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))
f = partial(optimized_function, X_scaled)

# ベイズ最適化を実行
study.optimize(f, n_trials = 150)

# 最適なパラメーターを選択
best_params = study.best_params


# In[326]:


kpca = KernelPCA(kernel = "rbf", **best_params, random_state = 0)
# kpca = KernelPCA(kernel = "rbf", gamma = 0.0001, random_state = 0)
X_kpca = kpca.fit_transform(X_scaled)
X_kpca_plane = X_kpca[:,:2]

k3n_error(X_scaled, X_kpca_plane)


# In[371]:


# プロット
x_0 = X_kpca[:, 0]
x_1 = X_kpca[:, 1]

#  図のサイズを指定
fig = plt.figure(figsize = ([8, 8]))
ax = fig.add_subplot(1,1,1)
plt.xlim([-2, 2])
plt.ylim([-2, 2])

#  フォントの調整
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20

# 軸名を指定
plt.xlabel("PC1")
plt.ylabel("PC2")

# 目盛の感覚を指定
plt.xticks(np.arange(-2, 3, 1))
plt.yticks(np.arange(-2, 3, 1))
    
# 各ラベルごとに色分け

scatter_color = plt.scatter(x_0, x_1, c = y / np.max(y), cmap = cm.hsv, s = 20)
plt.show()

# 各ラベルごとに色分け
# plt.scatter(x_0[y == 0], x_1[y == 0], c = "dodgerblue", s = 20)
# plt.scatter(x_0[y == 1], x_1[y == 1], c = "crimson", s = 20)
# plt.scatter(x_0[y == 2], x_1[y == 2], c = "limegreen", s = 20)
# plt.show()


# In[368]:


# プロット
x_0 = X_kpca[:, 0]
x_1 = X_kpca[:, 1]

#  図のサイズを指定

# plt.subplotsは配置に基づいたnumpy arrayを返すが、ImageGridはつねにリストを返す
grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.15, share_all=True, label_mode='L',
                cbar_location='right', cbar_mode='single')

fig.canvas.draw()


fig = plt.figure(figsize = ([8, 8]))
ax = fig.add_subplot(1,1,1)
plt.xlim([-2, 2])
plt.ylim([-2, 2])

#  フォントの調整
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20

# 軸名を指定
plt.xlabel("PC1")
plt.ylabel("PC2")

# 目盛の感覚を指定
plt.xticks(np.arange(-2, 3, 1))
plt.yticks(np.arange(-2, 3, 1))
    
# 各ラベルごとに色分け

scatter_color = plt.scatter(x_0, x_1, c = y / np.max(y), cmap = cm.hsv, s = 20)

axpos = ax.get_position()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad = 0.8)





# caxpos = cax.get_position()

# plt.colorbar(scatter_color, cax = cax)
    
# plt.show()

# 各ラベルごとに色分け
# plt.scatter(x_0[y == 0], x_1[y == 0], c = "dodgerblue", s = 20)
# plt.scatter(x_0[y == 1], x_1[y == 1], c = "crimson", s = 20)
# plt.scatter(x_0[y == 2], x_1[y == 2], c = "limegreen", s = 20)
# plt.show()


# In[ ]:


fig = plt.figure()
fig.suptitle('figsize=({}, {})'.format(fig.get_figwidth(), fig.get_figheight()))

# plt.subplotsは配置に基づいたnumpy arrayを返すが、ImageGridはつねにリストを返す
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.15, share_all=True, label_mode='L',
                cbar_location='right', cbar_mode='single')

fig.canvas.draw()

imgs = MultipleImagesAndRectangle(fig, grid, data)

cbar = grid.cbar_axes[0].colorbar(imgs[-1])
cbar.ax.axis('off') # これまでのcbar.axとはこのコマンドの振る舞いが少し違う（枠が消えてしまう）


# In[353]:


def MultipleImagesAndRectangle(fig, axes, data, draw_rect=True, aspect='equal') :
    imgs = []
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()

    vmax = len(axes)

    for i, ax in enumerate(axes):
        if draw_rect:
            AddAxesBBoxRectAndText(fig, ax, 'before imshow')

        imgs.append(ax.imshow((i+1)*data, origin='lower', vmin=0, vmax=vmax, aspect=aspect))
        ax.set_title('{}*data'.format(i+1))
        ax.axis('off')
    return imgs

