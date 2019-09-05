import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

x = np.array([-2, -1,  0, 2,  5,  8, 12]);
y = np.array([-8, -5, -2, 4, 13, 22, 34]);

model = keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
);

model.compile(
    optimizer = "sgd",
    loss = "mean_squared_error"
);

model.fit(x, y, epochs=500);

points = [];
for i in range(-50, 50):
    points.append(model.predict([i])[0, 0]);

plt.plot(points);
plt.xlabel('x');
plt.ylabel('y');
plt.show();