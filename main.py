from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import keras as keras
import numpy as np
import datetime as datetime
import sys as sys
from sklearn.preprocessing import MinMaxScaler

# Verify GPU
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# Data
# input structure: [minutes since midnight, likes, comments]
dataname = sys.argv[1] if len(sys.argv) > 1 != None else 'spacex.csv'
data = np.genfromtxt(dataname, delimiter=',')

x_train = data[1:, [0]]
y_train = data[1:, [1]]

# Normalize data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

# Tensorboard
tensorboardCallback = keras.callbacks.TensorBoard(log_dir='./graph/' + dataname + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")), histogram_freq=0,
                                                  write_graph=True, write_images=True)

# Model
model = Sequential()

# Input layer
model.add(Dense(256, input_dim=1, kernel_initializer='glorot_normal', activation='relu'))

# Hidden layers
model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu'))

# Output layer
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_absolute_error',
              optimizer=keras.optimizers.Adam(lr=0.005))

model.fit(x_train, y_train, batch_size=32, epochs=1000, shuffle=True,
          callbacks=[tensorboardCallback])

predicted = model.predict(x_train).astype(int)
print(np.column_stack((y_train, predicted)))    
