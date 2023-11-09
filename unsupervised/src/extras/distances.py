#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from itertools import product
#%%
def lp_norm(v_1: np.array, v_2: np.array, p: int):
    """Función que calcula la distancia lp
    v_1 (numpy array): vector al que se le va a calcular la distancia
    v_2 (numpy array): vector al que se le va a calcular la distancia
    p (int): potencia de la métrica lp
    
    Returns:
    vector_distancia (numpy array): con la distancia
    """
    return np.power(np.sum(np.abs(v_1 - v_2) ** p), 1/p)

def mahalanobis(v_1, v_2, cov_matrix):
    """Función que calcula la distancia mahalanobis 
    m_1 (np.matrix o np.array): matriz o vector a calcular la covarianza
    m_2 (np.matrix o np.array): matriz o vector a calcular la covarianza 
    
    Returns:
    np.array o float: vector o float con la distancia 
    """
    return np.dot(np.dot(v_1, np.linalg.inv(cov_matrix)), v_2)


def get_distances(data, distance, args=None):
    """Función que obtiene las distancias

    data (np.ndarray): Array multidimensional con los datos
    distance (str): distancia deseada (p, mahalanobis)
    
    Returns:
    distances (np.ndarray): Array multidimensional con los datos 
    """
    n1 = data.shape[0]
    distances = np.zeros((n1,n1))
    filas = data.shape[0]
    cov_matrix = np.cov(data, rowvar = True)
    p_distances = {}
    
    match distance:
        case "p":
            for p in args:
                p_distances[p] = np.zeros((n1,n1))
                for i,j in product(range(filas),range(filas)):
                    d = lp_norm(data[i,:], data[j,:], p)
                    p_distances[p][i,j] = d
            return p_distances
        case "mahalanobis":
            for i,j in product(range(filas),range(filas)):
                dif = data[i,:] - data[j,:]
                d = mahalanobis(dif, dif, cov_matrix)
            distances[i,j] = d

            return distances 






# %%
