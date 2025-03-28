import time
import pathlib

import numpy as np
import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as plt

this_path = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/data"
pathlib.Path(this_path).mkdir(parents=True, exist_ok=True)
data_file = pathlib.Path(this_path + "/mnist.npz")
data_model= pathlib.Path(this_path + "/model.mol")

tf.random.set_seed(time.time())
mnist = tf.keras.datasets.mnist
# print(f"{this_path}")
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=data_file)
x_train, x_test = x_train / 255.0, x_test / 255.0

# plt.imshow(x_train[0])
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss=tf.keras.losses.mse)
# model.fit(x_train, y_train, epochs=1)
# tf.keras.models.save_model(model=model, filepath=data_model, overwrite=True)

model = tf.keras.models.load_model(filepath=data_model, compile=False)
prediction = np.argmax(model.predict(np.reshape(x_test[5], (1, 28, 28))))
print(f"prediction: {prediction}")
predictions = model(x_train[:1]).numpy()
print(f"Prediction: {predictions}")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
rs_loss_fn = loss_fn(y_train[:1], predictions).numpy();
print(f"After loss fn: {rs_loss_fn}")

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()])
probability_model
print(f"Probability model: {probability_model(x_test[:5])}")