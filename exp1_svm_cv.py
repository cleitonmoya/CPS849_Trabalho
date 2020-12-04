# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:16:18 2020
@author: Cleiton Moya de Almeida

"""
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import  KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Experiment parameters
kernel = 'sigmoid'
K =  2      # k-fold parameter
test_size = 0.3

# Dataset reading
df = pd.read_table('banana.dat', skiprows=7, sep=',', names=['x1', 'x2', 'y'])
df = df.astype({'y': 'int32'})
X = df[['x1', 'x2']].to_numpy()
y = df.y.to_numpy()

# Data preprocessing
#min_max_scaler = MinMaxScaler()
#X = min_max_scaler.fit_transform(X) 

# Hyper-parameters range definition
#C_range = [0.01, 0.1, 1]
C_range = np.logspace(-3,3,7)
#C_range = np.logspace(-2,2,5)

#gamma_range = [0.01, 0.1, 1]
gamma_range = np.logspace(-3,3,7)
#gamma_range = np.logspace(-2,1,4)

#degree_range = [2, 3, 5]
coef0_range=[0, 0.5, 1]

#param_grid = [dict(C=C_range, gamma=gamma_range, degree=degree_range, coef0=coef0_range)]
param_grid = [dict(C=C_range, gamma=gamma_range, coef0=coef0_range)]
#param_grid = [dict(C=C_range, gamma=gamma_range)]

# Cross-validation
cv = KFold(n_splits=K)
grid = GridSearchCV(svm.SVC(kernel=kernel,cache_size=1000), param_grid=param_grid, cv=cv, n_jobs=-1,verbose=10)
grid.fit(X,y)
best_std_score = grid.cv_results_['std_test_score'][grid.best_index_]

# Results:
print("Best parameters: %s \nAccuracy: %0.3f \u00B1 %0.3f"
      % (grid.best_params_, grid.best_score_, best_std_score))