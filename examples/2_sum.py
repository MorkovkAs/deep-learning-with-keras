from keras.layers import Dense
from keras.models import Sequential
from numpy import array


def checksum(a, b):
    res_sum = model.predict(array([[a, b]]))[0][0]
    int_sum = int(round(res_sum))
    print("{0} + {1} = \t{2} \t({3})".format(a, b, int_sum, res_sum))


model = Sequential()
model.add(Dense(1, activation='linear'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

# Generate dummy data
import numpy as np

data = array([[np.random.randint(0, 10) for i in range(2)] for i in range(2000)])
labels = np.sum(data, axis=1)

# Train the model, iterating on the data in batches of 4 samples
model.fit(data, labels, epochs=10, batch_size=4)

checksum(4, 9)
checksum(8, 9)
checksum(39, 98)
checksum(-57, -7)
checksum(-4, 6)
checksum(41, 13)
checksum(0, 0)
