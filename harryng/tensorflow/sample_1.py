import time
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

this_path = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/data"
pathlib.Path(this_path).mkdir(parents=True, exist_ok=True)
data_file = pathlib.Path(this_path + "/mnist.npz")

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

predictions = model(x_train[:1]).numpy()
print(f"predictions: {predictions}")
softmax_predictions = tf.nn.softmax(predictions).numpy()
print(f"softmax prediction: {softmax_predictions}")
