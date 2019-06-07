from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import keras as keras
import numpy as np

# Verify GPU
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# Data
# input structure: [minutes since midnight, likes, comments]
data = np.genfromtxt('reknihy.csv', delimiter=',')
data = data.astype(int)

x_train = data[1:, [0]]
y_train = data[1:, [1]]

# Tensorboard
tensorboardCallback = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0,
                            write_graph=True, write_images=True)

# Model
model = Sequential()

model.add(Dense(128, input_dim=1, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=1, epochs=1000, callbacks=[tensorboardCallback])

predicted = model.predict(x_train).astype(int)
print(np.column_stack((y_train, predicted)))
