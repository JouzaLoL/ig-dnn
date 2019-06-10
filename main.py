from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import SGD
import keras as keras
import numpy as np
import datetime as datetime
import sys as sys
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

# Verify GPU
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# Data
# input structure: [minutes since midnight, likes, comments]
dataname = sys.argv[1] if len(sys.argv) > 1 != None else 'spacex.csv'
data = np.genfromtxt(dataname, delimiter=',')

# Fix data to minutes and discard first value
data[:, 0] = data[:, 0] * 100
data = data[1:]


# Sort data
data = data[data[:, 0].argsort()]

# Plot Raw training data
pyplot.subplot(211)
pyplot.title("Raw training data")
pyplot.plot(data[:, 0], data[:, 1], 'bo', label="likes")
pyplot.legend()
pyplot.subplot(212)
pyplot.plot(data[:, 0], data[:, 2], 'go', label="comments")
pyplot.legend()
pyplot.show()

# Split data into 10-minute clusters
# https://stackoverflow.com/questions/11767139/split-numpy-array-at-multiple-values
indices = np.array(range(0, 1440, 60))

split_at = data[:, 0].searchsorted(indices)
split_data = np.split(data, split_at)

data_avgs = []
for group in split_data:
    mean_values = np.mean(group, 0)
    data_avgs.append(mean_values)

# Throw out nan values
data_avgs = np.array(data_avgs)
data_avgs = data_avgs[~np.isnan(data_avgs).any(axis=1)]

# Plot grouped training data
pyplot.subplot(211)
pyplot.title("Grouped avg training data")
pyplot.plot(data_avgs[:, 0], data_avgs[:, 1], 'bo', label="likes")
pyplot.legend()
pyplot.subplot(212)
pyplot.plot(data_avgs[:, 0], data_avgs[:, 2], 'go', label="comments")
pyplot.legend()
pyplot.show()

# Extract x and y from data
x_train = data_avgs[1:, [0]]
y_train = data_avgs[1:, [1]]

# Normalize data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

pyplot.title("Averages normalized")
pyplot.plot(x_train, y_train, 'ro', label="train")
pyplot.legend()
pyplot.show()

# Tensorboard
tensorboardCallback = keras.callbacks.TensorBoard(log_dir='./graph/' + dataname + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")), histogram_freq=0,
                                                  write_graph=True, write_images=True)

# Model
model = Sequential()

# Input layer
model.add(Dense(64, input_dim=1,
                kernel_initializer='glorot_normal', activation='relu'))

# Hidden layers
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(), metrics=["mean_absolute_percentage_error", "mean_absolute_error"])

history = model.fit(x_train, y_train, batch_size=1, epochs=3000,
                    callbacks=[tensorboardCallback])

# plot loss during training
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.legend()
pyplot.show()

predicted = model.predict(x_train).astype(int)

pyplot.title("Predicted vs actual values")
pyplot.plot(x_train, predicted, 'bo', label="pred")
pyplot.plot(x_train, y_train, 'ro', label="train")
pyplot.legend()
pyplot.show()

print(np.column_stack((y_train, predicted)))
