# file for loading data to iris notebook
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load iris data
def load_data():
    iris = datasets.load_iris()
    x = iris.data[:, 2:]
    y = iris.target
    return x, y
