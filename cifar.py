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

    # Fetch test batch
    test_batch = unpickle(os.path.join(path, "test_batch"))
    x_test = test_batch[b"data"]
    y_test = np.array(test_batch[b"labels"])

    # Reshape data
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test  = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return x_train, y_train, x_test, y_test

def get_cifar10_label_names():
    meta = unpickle("cifar-10-batches-py/batches.meta")
    label_names = [name.decode("utf-8") for name in meta[b"label_names"]]
    return label_names

def show_examples(x_train, y_train):
    label_names = get_cifar10_label_names()

    plt.figure(figsize=(15, 15))
    
    for i in range(50):
        id = np.random.randint(0, len(x_train))
        img = x_train[id]
        label_num = y_train[id]
        label_name = label_names[label_num]

        plt.subplot(5, 10, i + 1)
        plt.imshow(img)
        plt.title(f"{label_num} - {label_name}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    x_train, y_train, x_test, y_test = load_cifar10()
    show_examples(x_train, y_train)

if __name__ == "__main__":
    main()

