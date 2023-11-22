#%%
from src.extras.distances import *
from src.extras.utils import normalizar_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
# %%
def create_centers(data: np.array, n_centers: int):
    centers = {}
    groups = {}
    for k in range(n_centers):
        idx = np.random.randint(len(iris_norm))
        centers[k] = iris_norm[idx]
        groups[k] = []
    return centers, groups

def k_means(data: np.array, n_iter: int, treshold: int, n_centers: int):
    centers, groups = create_centers(data, n_centers)
    J = float("inf")
    itr = 0
    while (J >= treshold) & (itr<= n_iter):
        _, groups = create_centers(data, n_centers)
        J = 0
        for x in range(len(data)):
            d_min = float("inf")
            for c in centers.keys():
                d_act = lp_norm(centers[c], data[x], 2)
                # Actualización de la pertenencia al grupo
                if d_act <= d_min:
                    d_min = d_act
                    c_act = c
            groups[c_act].append(data[x])
            J += d_min
        print(J)
        print(itr)
        itr += 1
        for c in groups.keys():
            # Dejar los grupos como un numpy.ndarray
            groups[c] = np.vstack(groups[c])
            centers[c] = np.mean(groups[c], axis=0)
    assert np.array([len(groups[i]) for i in groups.keys()]).sum() == len(data)
    return groups, centers


# %%
iris = datasets.load_iris()["data"]
iris_norm = normalizar_data(iris)
n_centers = 3
n_iter = 1000
treshold = 10
groups, centers = k_means(data = iris_norm, n_iter = n_iter, treshold = treshold, n_centers = n_centers)
