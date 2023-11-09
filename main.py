#%%
from unsupervised.src.extras.distances import get_distances
from sklearn import datasets
import seaborn as sns

def main():

    iris = datasets.load_iris()["data"]
    ds = get_distances(iris, "p", [1,2,3])
    sns.heatmap(ds[3])
    print("hola")
#%%
if __name__ == "__main__":
    main()
