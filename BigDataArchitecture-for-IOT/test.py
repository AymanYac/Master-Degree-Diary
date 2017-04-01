from sklearn.datasets import load_iris

from HoeffdingTree import HoeffdingTreeClassifier

def main():
    iris = load_iris()
    X = iris.data
    Y = iris.target

    ht = HoeffdingTreeClassifier()

    for t in range(0, len(X)):
        #ht.predict(X[t, :].reshape(1, -1))
        ht.partial_fit(X[t, :].reshape(1, -1), Y[t].reshape(1, -1))


if __name__ == '__main__':
    main()
