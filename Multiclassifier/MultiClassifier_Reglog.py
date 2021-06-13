import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1, out_p=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.out_p = out_p

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            output[(output == 1)] = 0.99999
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, self.out_p, -1), self.getProb(X)

    def getProb(self,X):
        return self.activation(self.net_input(X))


class MultiClassifier(object):

    def __init__(self, eta=0.05, n_iter=100, random_state=1, classes_no=2):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.classes_no=classes_no
        self.clsf = []
        for i in range(classes_no-1):
            self.clsf.append(LogisticRegressionGD(self.eta, self.n_iter, random_state,i))

    def fit(self, X, y, class_no):
        print(class_no)
        self.clsf[class_no].fit(X, y)

    def predict(self, X):
        pred = []
        for clf in self.clsf:
            predicted, prob = clf.predict(X)
            pred.append(predicted)
        res = pred[0].copy()
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

    def printProb(self, X, y, names):
        probs = []
        for clf in self.clsf:
            predicted, prob = clf.predict(X)
            probs.append(prob)
        for i in range(len(probs)):
            print(f'\nProbability for class {i}')
            for j in range(len(probs[i])):
                print('\t', X[j], names[y[j]], round(probs[i][j], 4))

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    train_y = []
    mcf = MultiClassifier(eta=0.1, n_iter=2000, random_state=1, classes_no=3)
    for tar in range(len(list(set(y))) - 1):
        tmp = y_train.copy()
        tmpX = X_train.copy()[(tmp == tar) | (tmp == tar + 1)]
        tmp = tmp[(tmp == tar) | (tmp == tar + 1)]
        tmp[(tmp != tar)] = -1
        tmp[(tmp == tar)] = 1
        tmp[(tmp == -1)] = 0
        mcf.fit(tmpX, tmp, tar)
        train_y.append(tmp)

    # print('==================================')
    # arr = mcf.predict(X_test)
    # for i in range(len(X_test)):
    #     print(X_test[i], y_test[i], arr[i], arr[i] == y_test[i])

    mcf.printProb(X_test, y_test, iris.target_names)

    plot_decision_regions(X=X_test, y=y_test, classifier=mcf)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()