import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

a = np.loadtxt('Sharp_char.txt')

startNo = 1
endNo = 16


def getAvgDist(y, cY):
    dist = y - cY
    return np.sum(abs(dist)) / len(y)


def getData(num):
    return np.loadtxt(f'Dane/dane{num}.txt')


for i in range(startNo, endNo):
    data = getData(i)

    x = data[:, [0]]
    y = data[:, [1]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1)

    x_range = np.linspace(np.amin(x), np.amax(x))

    c = np.hstack([X_train, np.ones(X_train.shape)])
    v = np.linalg.pinv(c) @ y_train
    y1 = v[0] * x_range + v[1]
    le = getAvgDist(v[0] * X_test + v[1], y_test)
    let = getAvgDist(v[0] * X_train + v[1], y_train)
    # print(let,v[0] * X_train + v[1],y_train)

    # print("lin", le, let)

    c1 = np.hstack([X_train ** 2, X_train, np.ones(X_train.shape)])
    v1 = np.linalg.pinv(c1) @ y_train
    y2 = v1[0] * x_range ** 2 + v1[1] * x_range + v1[2]
    qe = getAvgDist(v1[0] * X_test ** 2 + v1[1] * X_test + v1[2], y_test)
    qet = getAvgDist(v1[0] * X_train ** 2 + v1[1] * X_train + v1[2], y_train)

    # print("quad", qe, qet)

    c2 = np.hstack([X_train ** 3, X_train ** 2, X_train, np.ones(X_train.shape)])
    v2 = np.linalg.pinv(c2) @ y_train
    y3 = v2[0] * x_range ** 3 + v2[1] * x_range ** 2 + v2[2] * x_range + v2[3]
    ie = getAvgDist(v2[0] * X_test ** 3 + v2[1] * X_test ** 2 + v2[2] * X_test+ v2[3], y_test)
    iet = getAvgDist(v2[0] * X_train ** 3 + v2[1] * X_train ** 2 + v2[2] * X_train + v2[3], y_train)

    # print("cub", ie, iet)

    polyfit = 10
    z = np.polyfit(X_train[:, 0], y_train[:, 0], polyfit)
    f = np.poly1d(z)
    y4 = f(x_range)
    pe = getAvgDist(f(X_test), y_test)
    pet = getAvgDist(f(X_train), y_train)

    # print("ten",pe,pet)

    # ========== this also gives v
    # X_mean = np.mean(X_train)
    # Y_mean = np.mean(y_train)
    #
    # num = 0
    # den = 0
    #
    # for i in range(len(X_train)):
    #     num += (X_train[i] - X_mean)*(y_train[i] - Y_mean)
    #     den += (X_train[i] - X_mean)**2
    # a = num/den
    # b = Y_mean - a*X_mean
    # print(a,b)
    plt.suptitle(f'(R)Linear model: {round(abs(le), 4)} {round(abs(let), 4)}'
                 f'\n(G)Quadratic model: {round(abs(qe), 4)} {round(abs(qet), 4)}'
                 f'\n(B)Cubic model:{round(abs(ie), 4)} {round(abs(iet), 4)}'
                 f'\n(Y)Polynomial of {polyfit}th power model: {round(abs(pe), 4)} {round(abs(pet), 4)}')
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.plot(x_range, y1, 'r')
    plt.plot(x_range, y3, 'b')
    plt.plot(x_range, y2, 'g')
    plt.plot(x_range, y4, 'y')
    plt.show()
