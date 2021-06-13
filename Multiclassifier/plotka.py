from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # konfiguruje generator znaczników i mapę kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # rysuje wykres powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

        Z = Z.reshape(xx1.shape)

        cs = plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        cbar = plt.colorbar(cs)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # rysuje wykres wszystkich próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                        c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')

        # for idx, cl in enumerate(np.unique(y)):
        #     plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
        #                 s=100, alpha=0.4, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')

