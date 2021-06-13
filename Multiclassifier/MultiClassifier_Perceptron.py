import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10, p_out=1):
        self.eta = eta
        self.n_iter = n_iter
        self.p_out = p_out

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.learnPredict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, self.p_out, -1)

    def learnPredict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def printErrors(self):
        print(self.errors_)
        print(self.w_)


class MultiClassifier(object):

    def __init__(self, eta=0.01, n_iter=10, classes_no=3):
        self.eta = eta
        self.n_iter = n_iter
        self.classes_no = classes_no
        self.ppns = []
        for i in range(classes_no-1):
            self.ppns.append(Perceptron(self.eta, self.n_iter, i))

    def fit(self, X, y, ppn_index):
        print(ppn_index)
        self.ppns[ppn_index].fit(X, y)

    def predict(self, X):
        pred = []
        for ppn in self.ppns:
            pred.append(ppn.predict(X))
        res = pred[0].copy()
        # print(pred)
        for i in range(len(res)):
            if res[i] == -1:
                val = 0
                for j in range(len(pred)):
                    if pred[j][i] != -1:
                        res[i] = pred[j][i]
                        break
                    else:
                        val += 1
                if val == (self.classes_no - 1):
                    res[i] = self.classes_no - 1
        return res


def main():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    train_y = []
    mcf = MultiClassifier(eta=0.1, n_iter=500, classes_no=3)

    for tar in range(len(list(set(y))) - 1):
        tmp = y_train.copy()
        tmpX = X_train.copy()[(tmp == tar) | (tmp == tar + 1)]
        tmp = tmp[(tmp == tar) | (tmp == tar + 1)]
        tmp[(tmp != tar)] = -1
        tmp[(tmp == tar)] = 1
        mcf.fit(tmpX, tmp, tar)
        train_y.append(tmp)

    # arr = mcf.predict(X_test)
    # for i in range(len(X_test)):
    #     print(X_test[i], y_test[i], arr[i], arr[i] == y_test[i])

    plot_decision_regions(X=X_test, y=y_test, classifier=mcf)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    main()