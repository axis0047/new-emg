import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import random

import urllib3


def train():

    learning_rate = 0.003
    meta_step_size = 0.25

    inner_batch_size = 25
    eval_batch_size = 25

    meta_iters = 2000
    eval_iters = 5
    inner_iters = 4

    eval_interval = 1
    train_shots = 20
    shots = 5
    classes = 4


    '''
    Data Loader
    '''

    class CustomDataset:
        def __init__(self, root_dir, training, target_size=(28, 28)):
            self.root_dir = root_dir
            self.training = training
            self.data = []
            self.labels = []
            self.target_size = target_size

            split = "train" if training else "test"

            for class_idx, class_name in enumerate(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    self.labels.append(class_name)
                    for image_name in os.listdir(class_path):
                        image_path = os.path.join(class_path, image_name)
                        image = self.load_and_preprocess_image(image_path)
                        self.data.append((image, class_idx))

        def load_and_preprocess_image(self, image_path):
            image = Image.open(image_path)
            image = image.convert("L")
            image = image.resize(self.target_size, Image.LANCZOS)
            return np.array(image)

        def get_mini_dataset(self, batch_size, repetitions, shots, num_classes, split=False):
            temp_labels = np.zeros(shape=(num_classes * shots))
            temp_images = np.zeros(shape=(num_classes * shots, *self.target_size, 1))
            if split:
                test_labels = np.zeros(shape=(num_classes))
                test_images = np.zeros(shape=(num_classes, *self.target_size, 1))

            label_subset = random.sample(self.labels, num_classes)
            for class_idx, class_obj in enumerate(label_subset):
                class_data = [(image, label) for image, label in self.data if label == class_idx]
                temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
                if split:
                    if len(class_data) <= shots:
                        print(f"Warning: Not enough images for class {class_obj}. Skipping this class.")
                        continue
                    selected_images = random.sample(class_data, k=shots + 1)
                    test_labels[class_idx] = class_idx
                    test_images[class_idx] = np.expand_dims(selected_images[-1][0], axis=-1)
                    temp_images[class_idx * shots : (class_idx + 1) * shots] = np.expand_dims([image[0] for image in selected_images[:-1]], axis=-1)
                else:
                    if len(class_data) < shots:
                        print(f"Warning: Not enough images for class {class_obj}. Skipping this class.")
                        continue
                    temp_images[class_idx * shots : (class_idx + 1) * shots] = np.expand_dims([image[0] for image in random.sample(class_data, k=shots)], axis=-1)

            dataset = tf.data.Dataset.from_tensor_slices(
                (temp_images.astype(np.float32), temp_labels.astype(np.int32))
            )
            dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
            if split:
                return dataset, test_images, test_labels
            return dataset




    urllib3.disable_warnings()  # Disable SSL warnings that may happen during download.
    train_dataset = CustomDataset('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms',training=True)
    test_dataset = CustomDataset('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms',training=False)

    '''
    MODEL
    '''

    def conv_bn(x):
        x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)


    inputs = layers.Input(shape=(28, 28, 1))
    x = conv_bn(inputs)
    x = conv_bn(x)
    x = conv_bn(x)
    x = conv_bn(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile()
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    '''
    TRAIN
    '''

    training = []
    testing = []
    for meta_iter in range(meta_iters):
        frac_done = meta_iter / meta_iters
        cur_meta_step_size = (1 - frac_done) * meta_step_size
        # Temporarily save the weights from the model.
        old_vars = model.get_weights()
        # Get a sample from the full dataset.
        mini_dataset = train_dataset.get_mini_dataset(
            inner_batch_size, inner_iters, train_shots, classes
        )
        for images, labels in mini_dataset:
            with tf.GradientTape() as tape:
                preds = model(images)
                loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        new_vars = model.get_weights()
        # Perform SGD for the meta step.
        for var in range(len(new_vars)):
            new_vars[var] = old_vars[var] + (
                (new_vars[var] - old_vars[var]) * cur_meta_step_size
            )
        # After the meta-learning step, reload the newly-trained weights into the model.
        model.set_weights(new_vars)
        # Evaluation loop
        if meta_iter % eval_interval == 0:
            accuracies = []
            for dataset in (train_dataset, test_dataset):
                # Sample a mini dataset from the full dataset.
                train_set, test_images, test_labels = dataset.get_mini_dataset(
                    eval_batch_size, eval_iters, shots, classes, split=True
                )
                old_vars = model.get_weights()
                # Train on the samples and get the resulting accuracies.
                for images, labels in train_set:
                    with tf.GradientTape() as tape:
                        preds = model(images)
                        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                test_preds = model.predict(test_images)
                test_preds = tf.argmax(test_preds).numpy()
                num_correct = (test_preds == test_labels).sum()
                # Reset the weights after getting the evaluation accuracies.
                model.set_weights(old_vars)
                accuracies.append(num_correct / classes)
            training.append(accuracies[0])
            testing.append(accuracies[1])
            if meta_iter % 100 == 0:
                print(
                    "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
                )

    return model



