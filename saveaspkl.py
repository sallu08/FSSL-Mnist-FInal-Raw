
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pickle as pkl


Path = "C:/cp/Mnist_Png/Train/dataD/"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  Path,
  labels="inferred",
  label_mode="int",
  color_mode="grayscale",
  seed=123,
  shuffle=False,
  image_size=(28, 28))


train_ds1 = train_ds.unbatch()
images = np.asarray(list(train_ds1.map(lambda x, y: x)))
labels = np.asarray(list(train_ds1.map(lambda x, y: y)))

x_train, x_test, y_train, y_test = train_test_split( images, labels, train_size=0.75, random_state=9)
x_train=x_train / 255.0
x_test=x_test / 255.0

#to save it
with open("traind1.pkl", "wb") as f:
    pkl.dump([x_train, y_train], f)
with open("testd1.pkl", "wb") as f1:
    pkl.dump([x_test, y_test], f1)
