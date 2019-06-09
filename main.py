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
from matplotlib import pyplot

# Verify GPU
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# Data 
# input structure: [minutes since midnight, likes, comments]
dataname = sys.argv[1] if len(sys.argv) > 1 != None else 'spacex.csv'
data = np.genfromtxt(dataname, delimiter=',')

# Fix data to minutes and discard first value
data[:,0] = data[:,0] * 100
data = data[1:]


# Sort data
data = data[data[:,0].argsort()]

# Plot Raw training data
pyplot.subplot(211)
pyplot.title("Raw training data")
pyplot.plot(data[:,0], data[:,1], 'bo', label="likes")
pyplot.legend()
pyplot.subplot(212)
pyplot.plot(data[:,0], data[:,2], 'go', label="comments")
pyplot.legend()
pyplot.show()

# Split data into 10-minute clusters
# https://stackoverflow.com/questions/11767139/split-numpy-array-at-multiple-values
indices = np.array(range(0,1440, 60))
print(indices)

split_at = data[:, 0].searchsorted(indices)
split_data = np.split(data, split_at)

# Calculate averages
def Average(lst): 
    return sum(lst) / len(lst)

y_avgs = []
for row in y_train:
    y_avgs.append([Average(row)]) 
y_train = np.array(y_avgs)

x_avgs = []
for row in x_train:
    x_avgs.append([Average(row)]) 
x_train = x_avgs

# Normalize data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

# Extract x and y from data
x_train = data[1:, [0]]
y_train = data[1:, [1]]

# Fix bad data


pyplot.title("Averages transformed")
pyplot.plot(x_train, y_train, 'ro', label="train")
pyplot.legend()
pyplot.show()

# Tensorboard
tensorboardCallback = keras.callbacks.TensorBoard(log_dir='./graph/' + dataname + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")), histogram_freq=0,
                                                  write_graph=True, write_images=True)

# Model
model = Sequential()

# Input layer
model.add(Dense(512, input_dim=1,
                kernel_initializer='glorot_normal', activation='relu'))

# Hidden layers
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))

# Output layer
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(lr=0.05))

history = model.fit(x_train, y_train, batch_size=32, epochs=4000, shuffle=True,
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
