# -*- coding: utf-8 -*-

from absl import app
from absl import flags

import efficient_capsnet
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

def _get_mnist_dataset(num_classes: int = 10, scaling: bool = False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = tf.cast(tf.expand_dims(X_train, axis=-1), dtype=tf.float32)
    y_train = tf.one_hot(y_train, depth=num_classes, dtype=tf.float32)
    X_test = tf.cast(tf.expand_dims(X_test, axis=-1), dtype=tf.float32)
    y_test = tf.one_hot(y_test, depth=num_classes, dtype=tf.float32)

    if scaling is True:
        X_train = X_train / 255
        X_test = X_test / 255

    return (X_train, y_train), (X_test, y_test)


def main(_) -> None:
    param = efficient_capsnet.make_param()
    model = efficient_capsnet.make_model(param)
    mnist_train, mnist_test = _get_mnist_dataset(param.num_digit_caps,
                                                 0.3)
    X_train, y_train = mnist_train
    X_test, y_test = mnist_test

    checkpoint_dir = "checkpoints"
    initial_epoch = 0
    num_epochs = 10

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        checkpoints = [
            file for file in os.listdir(checkpoint_dir) if "ckpt" in file
        ]
        if len(checkpoints) != 0:
            checkpoints.sort()
            checkpoint_name = checkpoints[-1].split(".")[0]
            initial_epoch = int(checkpoint_name)
            model.load_weights(
                filepath=f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=f"{checkpoint_dir}/train_log.csv", append=True)
    model_saver = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir +
                                                     "/{epoch:04d}.ckpt",
                                                     save_weights_only=True)
    model.fit(x=X_train,
              y=y_train,
              validation_split=FLAGS.validation_split,
              initial_epoch=initial_epoch,
              epochs=initial_epoch + num_epochs,
              callbacks=[csv_logger, model_saver])
    model.save(f"{checkpoint_dir}/model")
    param.save_config(f"{checkpoint_dir}/config.txt")



if __name__ == "__main__":
    app.run(main)