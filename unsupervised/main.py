#%%
from sklearn import datasets
import seaborn as sns
from src.extras.utils import normalizar_data
from src.extras.unsupervised_algorithms import k_means
def main():
    iris = datasets.load_iris()["data"]
    iris_norm = normalizar_data(iris)
    n_centers = 3
    n_iter = 1000
    treshold = 10
    groups, centers = k_means(data = iris_norm, n_iter = n_iter, treshold = treshold, n_centers = n_centers)
#%%
if __name__ == "__main__":
    main()
