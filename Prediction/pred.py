# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = read_csv('ALIOR.mst', usecols=[2], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

dataset = dataset[1:60,:]
print('dataframe.shape =', dataframe.shape)

# normalize the dataset
ardataset = dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size-1:,:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network

model = Sequential()

model.add(LSTM(10, input_shape=(1, look_back)))
model.add(Dense(1))

'''
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(1))
'''


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('NN Train Score: %.2f' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('NN Test Score: %.2f' % (testScore))

# # custom test
#
# startTest = numpy.array(testX[0])
# plotting = numpy.array(startTest)
#
# # plotting = numpy.reshape(plotting,(1,1,1))
# print('start test ' , startTest, testX.shape ,plotting,plotting.shape)
# for i in range(0,test_size-1):
# 	# print(i,numpy.reshape(plotting[i],(1,1,1)))
# 	p = model.predict(numpy.reshape(plotting[i],(1,1,1)))
# 	pt = model.predict(numpy.reshape(testX[i],(1,1,1)))
# 	print(f'pred: {p} and {pt} from {plotting[i][0]} and {testX[i][0][0]}')
# 	plotting = numpy.append(plotting,numpy.reshape(p,(1,1)),axis=0)
#
# plotting = scaler.inverse_transform(plotting)


# train, test = ardataset[0:train_size,:], ardataset[train_size-1:,:]
# dataX, dataY = create_dataset(train,3)
# tdataX, tdataY = create_dataset(test,3)
x,y = create_dataset(ardataset,3)
dataX = x[0:train_size-4]
dataY = y[0:train_size-4]
tdataX = x[train_size-4:]
tdataY = y[train_size-4:]

v = numpy.linalg.pinv(dataX) @ dataY

trainScore = math.sqrt(mean_squared_error(dataY, v[0]*dataX[:,[0]] + v[1]*dataX[:,[1]] + v[2]*dataX[:,[2]]))
# trainScore = numpy.square(numpy.subtract(dataY,v[0]*dataX[:,[0]] + v[1]*dataX[:,[1]] + v[2]*dataX[:,[2]])).mean()
print('AR Train Score: %.2f' % (trainScore))
testScore = math.sqrt(mean_squared_error(tdataY, v[0]*tdataX[:,[0]] + v[1]*tdataX[:,[1]] + v[2]*tdataX[:,[2]]))
print('AR Test Score: %.2f' % (testScore))

# plot baseline and predictions
plt.scatter(range(len(dataset)),scaler.inverse_transform(dataset),label='dataset',marker='*')
plt.plot(range(len(dataset)),scaler.inverse_transform(dataset),label='dataset',alpha=0.3)
plt.plot(range(len(trainPredict)),trainPredict,label='train predict',alpha=0.7)
plt.plot(range(len(trainPredict) + 1,len(trainPredict) + len(testPredict)+1),testPredict,label='test predict', alpha=0.7)
# plt.plot(range(len(trainPredict) + 1,len(trainPredict) + len(plotting)+1),plotting,label='pure predict')
plt.plot(range(2,train_size-2),v[0]*dataX[:,[0]] + v[1]*dataX[:,[1]] + v[2]*dataX[:,[2]],label='ar train predict',alpha=0.7)
plt.plot(range(train_size-2,train_size-2 + test_size),v[0]*tdataX[:,[0]] + v[1]*tdataX[:,[1]] + v[2]*tdataX[:,[2]],label='ar test predict',alpha=0.7)
plt.legend()
plt.show()

