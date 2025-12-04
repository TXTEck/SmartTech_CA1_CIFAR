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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    x_train = reshape_images(x_train)
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
    x_train = reshape_images(x_train)
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

def normalize(img):
    img = img/255
    return img

def gaussian_blur(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = gaussian_blur(img)
    img = equalize(img)
    img = normalize(img)

    return img

def show_dataset_sizes(x_train, y_train, x_test, y_test):
    print("\nDataset Sizes")
    print(f"Training images: {len(x_train)}")
    print(f"Training labels: {len(y_train)}")
    print(f"Test images:     {len(x_test)}")
    print(f"Test labels:     {len(y_test)}")

def number_of_images(y, label_mapping=None, title=""):
    print(f"\n{title}")

    cifar10_names = get_cifar10_label_names()
    cifar100_names, _ = get_cifar100_label_names()

    unique, counts = np.unique(y, return_counts=True)

    for i in range(len(unique)):
        cls = unique[i]
        count = counts[i]

        orig_label = list(label_mapping.keys())[list(label_mapping.values()).index(cls)]

        if orig_label < 10:
            label_name = cifar10_names[orig_label]
        else:
            label_name = cifar100_names[orig_label]

        print(f"Class {cls:2d} ({label_name}): {count}")

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

    show_examples_cifar10(x_train_10,y_train_10)
    show_examples_cifar100(x_train_100,y_train_100)

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

def leNet_model(num_classes):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 1), activation='relu',))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(54, (5, 5), activation='relu',))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))

    model.add(Dense(84, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# improved model
#def improved_model(num_classes, input_shape=(32, 32, 1)):
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
        BatchNormalization, Activation, GlobalAveragePooling2D
    )

    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.30))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.35))

    # Classification head
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Evaluate model

def evaluate_model(model, x_train, y_train, x_test, y_test):
    print(model.summary())

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=30,
        batch_size=128,
        validation_data=(x_test, y_test),
        shuffle=True,
        verbose=1,
    )

    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.show()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", round(score[0]))
    print("Test accuracy:", round(score[1] * 100, 2), "%")

def main():
    x_train, y_train, x_test, y_test, label_mapping = load_and_merge_data()

    show_dataset_sizes(x_train, y_train, x_test, y_test)
    number_of_images(y_train, label_mapping, title="Training Set Class Distribution")
    number_of_images(y_test, label_mapping, title="Test Set Class Distribution")


    x_train = reshape_images(x_train)
    x_test  = reshape_images(x_test)

    # Preprocess
    x_train = np.array(list(map(preprocessing, x_train)))
    x_test  = np.array(list(map(preprocessing, x_test)))

    examine_random_image_after_processing(x_train)

    x_train, x_test = reshape_for_cnn(x_train, x_test)

    num_classes = len(label_mapping)
    y_train, y_test = one_hot_encode(y_train, y_test, num_classes)

    model = leNet_model(num_classes)
    evaluate_model(model, x_train, y_train, x_test, y_test)


#Mini VGG model for better performance

   # def mini_vgg_model(num_classes, input_shape=(32, 32, 1)):
      #  from tensorflow.keras.layers import (
       #     Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
        #    BatchNormalization, Activation
        #)

       # model = Sequential()

        # Block 1
        #model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        #model.add(BatchNormalization())
        #model.add(Activation("relu"))
        #model.add(Conv2D(32, (3, 3), padding="same"))
        #model.add(BatchNormalization())
       # model.add(Activation("relu"))
        #model.add(MaxPooling2D((2, 2)))
        #model.add(Dropout(0.25))

        # Block 2
        #model.add(Conv2D(64, (3, 3), padding="same"))
        #model.add(BatchNormalization())
        #model.add(Activation("relu"))
        #model.add(Conv2D(64, (3, 3), padding="same"))
        #model.add(BatchNormalization())
        #model.add(Activation("relu"))
        #model.add(MaxPooling2D((2, 2)))
        #model.add(Dropout(0.25))

        # Block 3
        #model.add(Conv2D(128, (3, 3), padding="same"))
        #model.add(BatchNormalization())
        #model.add(Activation("relu"))
        #model.add(Conv2D(128, (3, 3), padding="same"))
       # model.add(BatchNormalization())
       # model.add(Activation("relu"))
       # model.add(MaxPooling2D((2, 2)))
        #model.add(Dropout(0.25))

        # Classifier
       # model.add(Flatten())
        #model.add(Dense(512, activation="relu"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.5))
        #model.add(Dense(num_classes, activation="softmax"))

        #model.compile(
       #     optimizer=Adam(learning_rate=1e-3),
        #    loss="categorical_crossentropy",
       #     metrics=["accuracy"]
       # )

       # return model

if __name__ == "__main__":
    main()
