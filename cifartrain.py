import keras
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

import numpy as np
import matplotlib.pyplot as plt

#
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# encode labels
from keras.utils import np_utils

num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train, x_valid = x_train[5000:], x_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test_samples')
print(x_valid.shape[0], 'validation samples')

# model architecture smaller version of alexnet

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.summary()

# compile the model
# opt = model.optimizer(optimizer=keras.optimizers.Adam(lr=0.1))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics='accuracy')
#
# train model

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid),
                 callbacks=[checkpoint], verbose=2, shuffle=True, use_multiprocessing=True)

score= model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Accuracy: ', score[1])
