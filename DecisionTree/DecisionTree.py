import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from plotkab import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    maxDepth = 6
    for i in range(1, maxDepth):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=1)
        tree.fit(X_train, y_train)
        plot_decision_regions(X_test, y_test, classifier=tree, test_idx=None)
        plt.suptitle(f"Entropy; Depth={i}")
        plt.xlabel('Długość płatka [cm]')
        plt.ylabel('Szerokość płatka [cm]')
        plt.legend(loc='upper left')
        plt.show()

        tree = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=1)
        tree.fit(X_train, y_train)
        plot_decision_regions(X_test, y_test, classifier=tree, test_idx=None)
        plt.suptitle(f"Gini; Depth={i}")
        plt.xlabel('Długość płatka [cm]')
        plt.ylabel('Szerokość płatka [cm]')
        plt.legend(loc='upper left')
        plt.show()

    maxDepth = 15
    for i in range(maxDepth):
        forest = RandomForestClassifier(criterion='gini', n_estimators=i, random_state=1, n_jobs=2)
        forest.fit(X_train, y_train)
        plt.suptitle(f"Random forest gini; Estimators={i}")
        plot_decision_regions(X_test, y_test,classifier=forest, test_idx=None)
        plt.xlabel('Długość płatka [cm]')
        plt.ylabel('Szerokość płatka [cm]')
        plt.legend(loc='upper left')
        plt.show()



if __name__ == '__main__':
    main()
