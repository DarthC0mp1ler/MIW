import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def printTreeClassif(classification, minDepth,tableName, correctAnswers, xLabel):
    print('======================================================================\nTree classification:')
    plotP = [[], []]
    plotN = [[], []]
    for i in range(len(classification)):
        print(f'\tDepth = {minDepth + i}')
        print(f'\tCriterion = gini')
        cr = [0, 0]
        for j in range(len(classification[i][0])):
            if classification[i][0][j] == correctAnswers[j]:
                cr[0] += 1
            else:
                cr[1] += 1
        print(f'\t\tCorrect classifications: {cr[0]}, \n\t\tincorrect classifications: {cr[1]}')
        plotP[0].append(cr[0])
        plotN[0].append(cr[1])
        print(f'\tCriterion = entropy')
        cr = [0, 0]
        for j in range(len(classification[i][0])):
            if classification[i][0][j] == correctAnswers[j]:
                cr[0] += 1
            else:
                cr[1] += 1
        print(f'\t\tCorrect classifications: {cr[0]}, \n\t\tincorrect classifications: {cr[1]}')
        plotP[1].append(cr[0])
        plotN[1].append(cr[1])

    fig, plts = plt.subplots(2)
    fig.suptitle(tableName)
    fig.set_size_inches(6,6.5)
    plts[0].set_title('Correct classifications')
    plts[0].plot(range(minDepth,minDepth+len(plotP[0])), plotP[0],alpha=0.3,color='r')
    plts[0].plot(range(minDepth,minDepth+len(plotP[1])), plotP[1],alpha=0.3,color='b')
    plts[0].grid()

    plts[1].set_title('Incorrect classifications')
    plts[1].plot(range(minDepth,minDepth+len(plotN[0])), plotN[0],alpha=0.3,color='r')
    plts[1].plot(range(minDepth,minDepth+len(plotN[1])), plotN[1],alpha=0.3,color='b')
    plt.grid()
    plt.show()


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    criterionDict = {0: 'gini', 1: 'entropy'}
    correctClassification = []
    depthMin = 5
    depthMax = 10

    for i in range(depthMax - depthMin):
        classif = []
        for j in range(len(criterionDict.keys())):
            tre = DecisionTreeClassifier(criterion=criterionDict[j], max_depth=depthMin + i, random_state=1)
            tre.fit(X_train, y_train)
            classif.append(tre.predict(X))
            # tree.plot_tree(tre)
        correctClassification.append(classif)

    printTreeClassif(correctClassification, depthMin,'Decision trees', y, 'depth counter')

    minTrees = 1
    maxTrees = 20

    correctClassification = []

    for i in range(maxTrees - minTrees):
        classif = []
        for j in range(len(criterionDict.keys())):
            forest = RandomForestClassifier(criterion=criterionDict[j], n_estimators=minTrees + i, random_state=1,
                                            n_jobs=2)
            forest.fit(X_train, y_train)
            classif.append(forest.predict(X))
        correctClassification.append(classif)

    printTreeClassif(correctClassification, minTrees,'Random forest', y, 'number of trees')

if __name__ == '__main__':
    main()
