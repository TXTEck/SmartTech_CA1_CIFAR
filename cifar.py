import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

np.random.seed(0)


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

def load_cifar10():
    path = "cifar-10-batches-py"

    # Get training data
    x_train_list = []
    y_train_list = []

    for i in range(1, 6):
        batch = unpickle(os.path.join(path, f"data_batch_{i}"))
        x_train_list.append(batch[b"data"])
        y_train_list.extend(batch[b"labels"])

    # Stack all batches vertically
    x_train = np.vstack(x_train_list)
    y_train = np.array(y_train_list)

    # Fet test batch
    test_batch = unpickle(os.path.join(path, "test_batch"))
    x_test = test_batch[b"data"]
    y_test = np.array(test_batch[b"labels"])

    # Reshape data
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test  = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = load_cifar10()

if __name__ == "__main__":
    main()

