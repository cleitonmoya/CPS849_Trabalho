# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:16:18 2020
@author: Cleiton Moya de Almeida

"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data reading
df = pd.read_table('banana.dat', skiprows=7, sep=',', names=['x1', 'x2', 'y'])
df = df.astype({'y': 'int32'})
X = df[['x1', 'x2']].to_numpy()
y = df.y.to_numpy()

# 2. Scatter plot
plt.rcParams.update({'font.size': 8})
plt.figure(figsize=(4,3))
sns.scatterplot(data=df, x="x1", y="x2", hue=df.y.tolist(),s=5, palette='tab10',alpha=0.5,linewidth=0)
plt.tight_layout()

# 3. Descriptive anlysis
print(df.describe())

# 4. Classes distribution
print("Number of y==-1:", df[df.y==-1].y.count())
print("% of y==-1:", df[df.y==-1].y.count()/df.y.count())
print("Number of y==+1:", df[df.y==+1].y.count())
print("% of y==+11:", df[df.y==+1].y.count()/df.y.count())