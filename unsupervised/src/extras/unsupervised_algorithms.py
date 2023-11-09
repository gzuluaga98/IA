#%%
from distances import *
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
# %%


iris = datasets.load_iris()["data"]


# %%
n_centers = 3
centers = {}
groups = {}
for k in range(n_centers):
    idx = np.random.randint(len(iris))
    centers[k] = iris[idx]
    groups[k] = []

# Definir pertenencia a grupo
# %%
for x in range(len(iris)):
    d_min = 10**10
    for c in centers.keys():
        d_act = lp_norm(centers[c], iris[x], 2)
        # Actualización de la pertenencia al grupo
        if d_act <= d_min:
            d_min = d_act
            c_act = c
    groups[c_act].append(iris[x])
#%%
for c in groups.keys():
    # Dejar los grupos como un numpy.ndarray
    groups[c] = np.vstack(groups[c])
    centers[c] = np.mean(groups[c], axis=0)
        