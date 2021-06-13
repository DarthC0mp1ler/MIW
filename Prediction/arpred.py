import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('alior.txt')

# trim dataset
a = a[0:50]

test_size = (len(a)//10)*3
train_size = len(a) - test_size + 1
# whole datasets
y = a[:,[0]]
c = a[:,1:]
# splitting
X_train = y[1:train_size]
y_train = c[1:train_size]
X_test = y[train_size:]
y_test = c[train_size:]

# Linear regression
v = np.linalg.pinv(y_train) @ X_train

#
print(y_train[:,[0]],y_train[:,[1]],y_train[:,[2]])

plt.plot(range(len(y)),y,'b-')
plt.plot(range(train_size-1),v[0]*y_train[:,[0]] + v[1]*y_train[:,[1]] + v[2]*y_train[:,[2]],'r')
plt.plot(range(train_size-1,train_size+test_size - 2),v[0]*y_test[:,[0]] + v[1]*y_test[:,[1]] + v[2]*y_test[:,[2]],'y')
plt.show()

