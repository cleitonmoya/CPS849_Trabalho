# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:16:18 2020
@author: Cleiton Moya de Almeida

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import  KFold, GridSearchCV
import graphviz 

# Dataset reading
df = pd.read_table('banana.dat', skiprows=7, sep=',', names=['x1', 'x2', 'y'])
df = df.astype({'y': 'int32'})
X = df[['x1', 'x2']].to_numpy()
y = df.y.to_numpy()

param_grid = [dict(criterion = ['gini', 'entropy'],
                   min_impurity_decrease = [0, 0.01, 0.05, 0.1],
                   max_depth = [3, 4, 5],
                   max_leaf_nodes = [3, 5])]

# Cross-validation
K = 10 # k-fold parameter
cv = KFold(n_splits=K)
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=cv, n_jobs=-1,verbose=10)
grid.fit(X,y)
best_std_score = grid.cv_results_['std_test_score'][grid.best_index_]

# Results:
print("Best parameters: %s \nAccuracy: %0.3f \u00B1 %0.3f"
      % (grid.best_params_, grid.best_score_, best_std_score))

# Model
clf = DecisionTreeClassifier(min_impurity_decrease=0.01, 
                             max_depth = 3,
                             min_samples_leaf = 1,
                             max_leaf_nodes = 3
                             )
clf = clf.fit(X,y)


dot_data = export_graphviz(clf, out_file=None,
                           feature_names=['x1', 'x2'],
                           class_names=['-1', '+1'],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data) 
graph.render("exp1-tree") 


# Decision surface
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(4,3))
plot_step = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

plot_colors = ['C0','C1']
cmap = matplotlib.colors.ListedColormap(plot_colors)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=cmap,alpha=0.5)

labels=['-1', '+1']

n_classes = 2
for i, color in zip([-1,1], plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=labels[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=10,linewidths=0.2)

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.tight_layout()