# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:16:18 2020
# Experimento 1 - KEEL Dataset
# Aplicação SVM - RBF
@author: cleiton

"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm

# Experiment parameters
test_size = 0.3     # tamanho do conjunto de testes

# Dataset reading
df = pd.read_table('banana.dat', skiprows=7, sep=',', names=['x1', 'x2', 'y'])
df = df.astype({'y': 'int32'})
X = df[['x1', 'x2']].to_numpy()
y = df.y.to_numpy()


# Scatter
colors=['C0', 'C1']
cmap = matplotlib.colors.ListedColormap(colors)
plt.scatter(X[:, 0], X[:, 1], c=y, s=1,cmap=cmap)

#clf = svm.SVC(kernel='linear',C=0.01,gamma=0.01)
clf = svm.SVC()
clf.fit(X,y)

# Plotting
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(4,3))

# Decision function plot
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10,
           linewidth=1, facecolors='none', edgecolors='k')
plt.tight_layout()

