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

    # Required CIFAR-10 classes
    SELECTED_LABELS = [1, 2, 3, 4, 5, 7, 9]

    x_train_list = []
    y_train_list = []

    # Load training batches
    for i in range(1, 6):
        batch = unpickle(os.path.join(path, f"data_batch_{i}"))
        x_batch = batch[b"data"]
        y_batch = np.array(batch[b"labels"])

        # Filter required labels
        mask = np.isin(y_batch, SELECTED_LABELS)
        x_train_list.append(x_batch[mask])
        y_train_list.extend(y_batch[mask])

    # Combine training data
    x_train = np.vstack(x_train_list)
    y_train = np.array(y_train_list)

    # Load test batch
    test_batch = unpickle(os.path.join(path, "test_batch"))
    x_test_raw = test_batch[b"data"]
    y_test_raw = np.array(test_batch[b"labels"])

    # Filter test batch
    filter = np.isin(y_test_raw, SELECTED_LABELS)
    x_test = x_test_raw[filter]
    y_test = y_test_raw[filter]

    return x_train, y_train, x_test, y_test

#Load CIFAR100 data
def load_cifar100():
    path = "cifar-100-python"

    SELECTED_FINE_LABELS = [
        19, 34, 2, 11, 35, 46, 98, 65, 80,     
        47, 52, 56, 59, 96,                   
        8, 13, 48, 58, 90, 41, 89             
    ]

    train = unpickle(os.path.join(path, "train"))
    test  = unpickle(os.path.join(path, "test"))

    x_train_raw = train[b"data"]
    y_fine_train_raw = np.array(train[b"fine_labels"])

    x_test_raw = test[b"data"]
    y_fine_test_raw = np.array(test[b"fine_labels"])

    # Filter
    filter = np.isin(y_fine_train_raw, SELECTED_FINE_LABELS)
    x_train = x_train_raw[filter]
    y_train = y_fine_train_raw[filter]

    filter = np.isin(y_fine_test_raw, SELECTED_FINE_LABELS)
    x_test = x_test_raw[filter]
    y_test = y_fine_test_raw[filter]

    return x_train, y_train, x_test, y_test

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
        id = np.random.randint(0, len(x_train))
        img = x_train[id]
        label = y_train[id]
        name = label_names[label]

        plt.subplot(5, 10, i + 1)
        plt.imshow(img)
        plt.title(f"{label} - {name}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#Show CIFAR100 examples
def show_examples_cifar100(x_train, y_train):
    fine_names, _ = get_cifar100_label_names()

    plt.figure(figsize=(15, 15))

    for i in range(50):
        id = np.random.randint(0, len(x_train))
        img = x_train[id]

        fine_label = y_train[id]
        fine_name = fine_names[fine_label]

        plt.subplot(5, 10, i + 1)
        plt.imshow(img)
        plt.title(f"{fine_label} - {fine_name}", fontsize=7)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def reshape_images(x):
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

def merge_data(
    x_train_10, y_train_10,
    x_train_100, y_train_100,
    x_test_10, y_test_10,
    x_test_100, y_test_100
):
    
    X_train = np.vstack([x_train_10, x_train_100])
    y_train = np.hstack([y_train_10, y_train_100])

    
    unique_labels = np.sort(np.unique(y_train))

    label_mapping = {}
    new_id = 0
    for orig in unique_labels:
        label_mapping[orig] = new_id
        new_id += 1

    
    y_train_mapped = np.array([label_mapping[label] for label in y_train])

    
    X_test = np.vstack([x_test_10, x_test_100])
    y_test = np.hstack([y_test_10, y_test_100])

    
    y_test_mapped = np.array([label_mapping[label] for label in y_test])

    return X_train, y_train_mapped, X_test, y_test_mapped, label_mapping


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def examine_random_image_after_processing(x_train):
    img = x_train[np.random.randint(0, len(x_train))]
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()


def load_and_merge_data():
    # CIFAR-10 
    x_train_10, y_train_10, x_test_10, y_test_10 = load_cifar10()

    # CIFAR-100 
    x_train_100, y_train_100, x_test_100, y_test_100 = load_cifar100()

    # Merge
    x_train, y_train, x_test, y_test, label_mapping = merge_data(
        x_train_10, y_train_10,
        x_train_100, y_train_100,
        x_test_10, y_test_10,
        x_test_100, y_test_100
    )

    return x_train, y_train, x_test, y_test, label_mapping

def reshape_for_cnn(x_train,x_test):
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    return x_train, x_test

def one_hot_encode(y_train, y_test, num_classes):
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return y_train, y_test


def main():
    x_train, y_train, x_test, y_test, label_mapping = load_and_merge_data()

    x_train = reshape_images(x_train)
    x_test  = reshape_images(x_test)

    # Preprocess
    x_train = np.array(list(map(preprocessing, x_train)))
    x_test  = np.array(list(map(preprocessing, x_test)))

    examine_random_image_after_processing(x_train)

    x_train, x_test = reshape_for_cnn(x_train, x_test)

    num_classes = len(label_mapping)
    one_hot_encode(y_train, y_test, num_classes)

    print(num_classes)





if __name__ == "__main__":
    main()
