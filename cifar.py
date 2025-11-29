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

#Unpickle data
def unpickle(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data

#Load CIFAR10 data
def load_cifar10():
    path = "cifar-10-batches-py"

    x_train_list = []
    y_train_list = []

    # Load training batches
    for i in range(1, 6):
        batch = unpickle(os.path.join(path, f"data_batch_{i}"))
        x_train_list.append(batch[b"data"])
        y_train_list.extend(batch[b"labels"])

    # Combine training data
    x_train = np.vstack(x_train_list)
    y_train = np.array(y_train_list)

    # Load test batch
    test_batch = unpickle(os.path.join(path, "test_batch"))
    x_test = test_batch[b"data"]
    y_test = np.array(test_batch[b"labels"])

    return x_train, y_train, x_test, y_test

#Load CIFAR100 data
def load_cifar100():
    path = "cifar-100-python"

    train = unpickle(os.path.join(path, "train"))
    test  = unpickle(os.path.join(path, "test"))

    x_train = train[b"data"]
    y_fine_train = np.array(train[b"fine_labels"])
    y_coarse_train = np.array(train[b"coarse_labels"])

    x_test = test[b"data"]
    y_fine_test = np.array(test[b"fine_labels"])
    y_coarse_test = np.array(test[b"coarse_labels"])

    return x_train, y_fine_train, y_coarse_train, x_test, y_fine_test, y_coarse_test

#Get CIFAR-10 label names
def get_cifar10_label_names():
    meta = unpickle("cifar-10-batches-py/batches.meta")
    label_names = [name.decode("utf-8") for name in meta[b"label_names"]]
    return label_names

#Get CIFAR-100 fine label names
def get_cifar100_label_names():
    meta = unpickle("cifar-100-python/meta")
    fine_label = [name.decode("utf-8") for name in meta[b"fine_label_names"]]
    coarse_label = [name.decode("utf-8") for name in meta[b"coarse_label_names"]]

    return fine_label, coarse_label

#Show CIFAR10 examples
def show_examples_cifar10(x_train, y_train):
    label_names = get_cifar10_label_names()

    plt.figure(figsize=(15, 15))

    for i in range(50):
        idx = np.random.randint(0, len(x_train))
        img = x_train[idx]
        label = y_train[idx]
        name = label_names[label]

        plt.subplot(5, 10, i + 1)
        plt.imshow(img)
        plt.title(f"{label} - {name}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#Show CIFAR100 examples
def show_examples_cifar100(x_train, y_fine_train, y_coarse_train):
    fine_names, coarse_names = get_cifar100_label_names()

    plt.figure(figsize=(15, 15))

    for i in range(50):
        idx = np.random.randint(0, len(x_train))
        img = x_train[idx]

        fine_label = y_fine_train[idx]
        coarse_label = y_coarse_train[idx]

        fine_name = fine_names[fine_label]
        coarse_name = coarse_names[coarse_label]

        plt.subplot(5, 10, i + 1)
        plt.imshow(img)
        plt.title(f"{fine_label} {fine_name}\n({coarse_label} {coarse_name})",
                  fontsize=7)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def reshape_images(x):
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
def main():

    # CIFAR-10
    x_train_10, y_train_10, x_test_10, y_test_10 = load_cifar10()
    x_train_10 = reshape_images(x_train_10)
    show_examples_cifar10(x_train_10, y_train_10)

    # CIFAR-100
    x_train_100, y_fine_train_100, y_coarse_train_100, x_test_100, y_fine_test_100, y_coarse_test_100 = load_cifar100()
    x_train_100 = reshape_images(x_train_100)
    show_examples_cifar100(x_train_100, y_fine_train_100, y_coarse_train_100)

if __name__ == "__main__":
    main()
