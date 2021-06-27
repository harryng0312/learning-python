# import keras
# from keras.models import Sequential
# from keras.layers import *
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# from keras.datasets import mnist

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pathlib

this_path = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/data"
pathlib.Path(this_path).mkdir(parents=True, exist_ok=True)
data_file = pathlib.Path(this_path + "/mnist.npz")
data_model = pathlib.Path(this_path + "/model_2.mol")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=data_file)
x_train, x_test = x_train / 255.0, x_test / 255.0

# extract label vector to matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(f"")

# create the sequence
model = Sequential()
# create input layer
model.add(Flatten(input_shape=[28, 28]))
# model.add(Dense(output_dim=256, activation=tf.keras.activations.relu, units=256))
# model.add(Dense(output_dim=128, activation=tf.keras.activations.relu, units=128))
# model.add(Dense(output_dim=10, activation=tf.keras.activations.softmax, units=10))
model.add(Dense(activation=tf.keras.activations.relu, units=256))
model.add(Dense(activation=tf.keras.activations.relu, units=128))
# create output class
model.add(Dense(activation=tf.keras.activations.softmax, units=10))
model.compile(loss=tf.keras.losses.categorical_crossentropy,  #loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
model.summary()

# traing model
# model.fit(x_train, y_train, epochs=10, verbose=2)
# tf.keras.models.save_model(model=model, filepath=data_model)s

# eval model
# score = model.evaluate(x_test, y_test, verbose=0)
# print(score)

# prediction
model = tf.keras.models.load_model(filepath=data_model)
plt.imshow(x_test[1999].reshape(28, 28))
plt.show()
y_predict = np.argmax(model.predict(x_test[1999].reshape(1, 28, 28, 1)))
print('Giá trị dự đoán: ', y_predict)