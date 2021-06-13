from numpy import argmax
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

def modelOneLayer():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modelTwoLayers():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modelThreeLayers():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

models = []
create = False
for i in range(3):
    try:
        model = load_model(f'model{i}')
        models.append(model)
    except OSError:
        create = True
        continue

(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))
testX = testX.reshape((testX.shape[0], 32, 32, 3))
trainY = to_categorical(trainY)
testY = to_categorical(testY)

if create:

    print('creating models')
    models = []

    trainX = trainX.astype('float32')/255.0
    testX = testX.astype('float32')/255.0

    models.append(modelOneLayer())
    models.append(modelTwoLayers())
    models.append(modelThreeLayers())
    i = 0
    for m in models:
        m.fit(trainX, trainY,epochs=10, batch_size=64, validation_data=(testX, testY))
        m.save(f'model{i}')
        i += 1

for m in models:
    loss, acc = m.evaluate(testX, testY)
    # Visualization
    numberOfTests = 4
    fig, plt = pyplot.subplots(numberOfTests, numberOfTests, constrained_layout=True, sharey=True)
    pyplot.suptitle(f'Acc: {round(acc * 100.0,4)}%')
    plt = plt.reshape(numberOfTests*numberOfTests)

    for i in range(len(testX[:(numberOfTests*numberOfTests)])):
        dtx = testX[i:i+1]
        d = m.predict(dtx)
        md = argmax(d[0])
        plt[i].imshow(testX[i], cmap=pyplot.get_cmap('gray'))
        plt[i].set_title(f'Prd: {md}, corr: {argmax(testY[i])}')
    pyplot.show()





