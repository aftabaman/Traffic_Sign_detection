

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random

with open("train.p", mode="rb") as training_data:
    train = pickle.load(training_data)

with open("valid.p", mode="rb") as validation_data:
    valid = pickle.load(validation_data)

with open("test.p", mode="rb") as testing_data:
    test = pickle.load(testing_data)

x_train, y_train = train["features"], train["labels"]

x_test, y_test = test["features"], test["labels"]

x_valid, y_valid = valid["features"], valid["labels"]

# data visualization

rand = np.random.randint(1, len(x_train))

plt.imshow(x_train[rand])

x_grid = 10
y_grid = 10

fig, axes = plt.subplots(x_grid, y_grid, figsize=(10, 10))

axes = axes.ravel()

n_train = len(x_train)

for i in range(x_grid * y_grid):
    index = np.random.randint(1, n_train)

    axes[i].imshow(x_train[index])

    axes[i].set_title(y_train[index], fontsize=20)

plt.subplots_adjust(hspace=1.5)

# shufflinfg the data

from sklearn.utils import shuffle

x_train, y_train = shuffle(x_train, y_train)

# turning the image from normal to grey image

x_train_gray = np.sum(x_train / 3, axis=3, keepdims=True)
x_test_gray = np.sum(x_test / 3, axis=3, keepdims=True)
x_valid_gray = np.sum(x_valid / 3, axis=3, keepdims=True)

x_train_gray.shape

# normalizin the images to be fitted in  to the model

x_train_gray_norm = (x_train_gray - 182) / 182
x_test_gray_norm = (x_test_gray - 182) / 182
x_valid_gray_norm = (x_valid_gray - 182) / 182

x_train_gray_norm

i = random.randint(1, len(x_train_gray))
plt.imshow(x_train_gray[i].squeeze(), cmap='gray')
plt.figure()
plt.imshow(x_train[i])
plt.figure()
plt.imshow(x_train_gray_norm[i].squeeze(), cmap='gray')

from tensorflow.keras import datasets, layers, models

CNN = models.Sequential()

CNN.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))

CNN.add(layers.AveragePooling2D())

CNN.add(layers.Dropout(0.2))

CNN.add(layers.Conv2D(16, (5, 5), activation='relu'))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Flatten())
CNN.add(layers.Dense(120, activation='relu'))
CNN.add(layers.Dense(84, activation='relu'))
CNN.add(layers.Dense(43, activation='softmax'))

CNN.summary()

CNN.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = CNN.fit(x_train_gray_norm,
                  y_train,
                  batch_size=300,
                  epochs=15,
                  verbose=1,
                  validation_data=(x_valid_gray_norm, y_valid))

score = CNN.evaluate(x_test_gray_norm, y_test)
print("Test Accuracy {}", format(score[1]))

history.history.keys()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, loss, 'ro', label='training_loss')
plt.plot(epochs, val_loss, 'r', label='validation_loss')
plt.title('Training and validation loss')

plt.plot(epochs, accuracy, 'b', label='training_accuracy')
plt.plot(epochs, val_accuracy, 'r', label='validation_accuracy')
plt.title('Training and validation accuracy')

predicted_classes = CNN.predict_classes(x_test_gray_norm)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True)

L = 10
W = 10

fig, axes = plt.subplots(L, W, figsize=(15, 15))

axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(x_test[i])
    axes[i].set_title('predection-{}\n True -{}'.format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')

    plt.subplots_adjust(wspace=1)

from matplotlib import image
from PIL import Image

image = image.imread('11_right-of-way.jpg')
plt.imshow(image)
plt.show()
data = np.asarray(image)
print(data.shape)
load_img_rz = np.array(
    Image.open('11_right-of-way.jpg').resize(
        (32, 32)))

plt.imshow(load_img_rz)
print("After resizing:", load_img_rz.shape)

load_img_rz_gray = np.sum(load_img_rz / 3, axis=2, keepdims=True)

# normalizin the images to be fitted in  to the model

load_img_rz_gray_norm = (load_img_rz_gray - 182) / 182
load_img_rz_gray_norm.shape

load_img_rz_gray_norm = tf.expand_dims(load_img_rz_gray_norm, 0)

predict = CNN.predict_classes(load_img_rz_gray_norm)

names = pd.read_csv("signnames.csv")
df_contains = names.ClassId.isin(predict)

# this means the image is in the names file

if np.any(df_contains.values):
    indx = names[names['ClassId'] == 11].index.values
    value = names.loc[indx].values

    plt.imshow(image)
    plt.title('{}'.format(value[-1][-1]))
    plt.axis("off")

else:
    print("Unkknown Sign!!")