import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('Sharp_char.txt')

x = a[:,[1]]
y = a[:,[0]]

c = np.hstack([x, np.ones(x.shape)]) # model 1 liniowy
c1 = np.hstack([x**2, x, np.ones(x.shape)]) # model kwadratowy
c2 = np.hstack([1/x, np.ones(x.shape)]) # model inny


v = np.linalg.pinv(c) @ y
v1 = np.linalg.pinv(c1) @ y
v2 = np.linalg.inv(c2.T @ c2) @ c2.T @ y

e = sum((y - (v[0]*x + v[1]))**2)

e1 = sum((y - (v1[0]*x**2 + v1[1]*x + v1[2]))**2)

print(e)


plt.plot(x, y, 'ro')
plt.plot(x,v[0]*x + v[1])
plt.plot(x,v1[0]*x**2 + v1[1]*x + v1[2])
plt.plot(x,v2[0]/x + v2[1])
plt.show()

