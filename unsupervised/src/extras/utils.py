import numpy as np

def normalizar(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min)/(x_max - x_min)
    return x_norm

def normalizar_data(x):
    data_norm = np.empty_like(x)
    for i in range(x.shape[1]):
        x_norm = normalizar(x[:,i])
        data_norm[:,i] = x_norm
    return data_norm