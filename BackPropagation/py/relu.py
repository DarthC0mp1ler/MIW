import numpy as np
import matplotlib.pyplot as plt

from time import sleep

P = np.arange(-4, 4.1, 0.1)

P = P.reshape(1,len(P))

T = P ** 2 + (np.random.rand(1,len(P[0,:])) - 0.5)

S1 = 100
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5
lr = 0.001

plt.ion()
fig = plt.figure()
plt1 = fig.add_subplot(1,1,1)

line, = plt1.plot(P[0,:],np.zeros(len(P[0,:])),'r+')
plt1.scatter(P,T,alpha=0.5,marker='*')

epochsNo = 100

for epoch in range(epochsNo):

    X = W1 @ P + B1 @ np.ones(P.shape)
    A1 = np.maximum(X, np.zeros(X.shape))
    A2 = W2 @ A1 + B2

    E2 = T - A2
    E1 = W2.T @ E2

    dW2 = lr * E2 @ A1.T
    dB2 = lr * E2 @ np.ones(E2.shape).T
    dW1 = lr * (np.exp(X) / (np.exp(X) + 1)) * E1 @ P.T
    dB1 = lr * (np.exp(X)/(np.exp(X)+1)) * E1 @ np.ones(P.shape).T

    W2 = W2 + dW2
    B2 = B2 + dB2
    W1 = W1 + dW1
    B1 = B1 + dB1

    if epoch % 3 == 0:
        A2s = A2.copy()
        fig.canvas.flush_events()
        line.set_ydata(A2s.reshape(A2.shape[1]))
        fig.canvas.draw()
        sleep(0.1)

plt.suptitle(f"Terminated, {epochsNo}epochs")
plt.ioff()
plt.show()