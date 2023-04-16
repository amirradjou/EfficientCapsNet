from absl import app
from absl import flags

import efficient_capsnet
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
'''
This is the main file for training and testing the Efficient CapsNet model.
In this file, we define the flags for the training and testing process.
We also define the functions for loading the MNIST dataset and plotting the training logs.
'''

import os

class CapsNetParam(object):
    __slots__ = [
        "input_width","input_height","input_channel","conv1_filter","conv1_kernel","conv1_stride","conv2_filter","conv2_kernel","conv2_stride","conv3_filter","conv3_kernel","conv3_stride","conv4_filter","conv4_kernel","conv4_stride","dconv_filter","dconv_kernel","dconv_stride","num_primary_caps","dim_primary_caps","num_digit_caps","dim_digit_caps",
    ]

    def __init__(self,input_width: int = 28,input_height: int = 28,input_channel: int = 1,conv1_filter: int = 32,conv1_kernel: int = 5,conv1_stride: int = 1,conv2_filter: int = 64,conv2_kernel: int = 3,conv2_stride: int = 1,conv3_filter: int = 64,conv3_kernel: int = 3,conv3_stride: int = 1,conv4_filter: int = 128,conv4_kernel: int = 3,conv4_stride: int = 2,dconv_kernel: int = 9,dconv_stride: int = 1,num_primary_caps: int = 16,dim_primary_caps: int = 8,num_digit_caps: int = 10,dim_digit_caps: int = 16,*args,**kwargs) -> None:


        # PrimaryCap Layer
        self.dconv_filter = num_primary_caps * dim_primary_caps
        self.dconv_kernel = dconv_kernel
        self.dconv_stride = dconv_stride
        self.num_primary_caps = num_primary_caps
        self.dim_primary_caps = dim_primary_caps

        # DigitCap Layer
        self.num_digit_caps = num_digit_caps
        self.dim_digit_caps = dim_digit_caps


        # Input Specification
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel

        # FeatureMap Layer
        self.conv1_filter = conv1_filter
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.conv2_filter = conv2_filter
        self.conv2_kernel = conv2_kernel
        self.conv2_stride = conv2_stride
        self.conv3_filter = conv3_filter
        self.conv3_kernel = conv3_kernel
        self.conv3_stride = conv3_stride
        self.conv4_filter = conv4_filter
        self.conv4_kernel = conv4_kernel
        self.conv4_stride = conv4_stride



    def get_config(self) -> dict:
        return {"input_width": self.input_width,"input_height": self.input_height,"input_channel": self.input_channel,"conv1_filter": self.conv1_filter,"conv1_kernel": self.conv1_kernel,"conv1_stride": self.conv1_stride,"conv2_filter": self.conv2_filter,"conv2_kernel": self.conv2_kernel,"conv2_stride": self.conv2_stride,"conv3_filter": self.conv3_filter,"conv3_kernel": self.conv3_kernel,"conv3_stride": self.conv3_stride,"conv4_filter": self.conv4_filter,"conv4_kernel": self.conv4_kernel,"conv4_stride": self.conv4_stride,"dconv_filter": self.dconv_filter,"dconv_kernel": self.dconv_kernel,"dconv_stride": self.dconv_stride,"num_primary_caps": self.num_primary_caps,"dim_primary_caps": self.dim_primary_caps,"num_digit_caps": self.num_digit_caps,"dim_digit_caps": self.dim_digit_caps
        }

    def save_config(self, path: str) -> None:

        if not isinstance(path, str):
            raise TypeError()
        elif len(path) == 0:
            raise ValueError()
        else:
            with open(path, 'w', encoding='utf8') as f:
                for k, v in self.get_config().items():
                    f.writelines(f"{k}={v}\n")


def load_config(path: str) -> CapsNetParam:
    if not isinstance(path, str):
        raise TypeError()
    elif len(path) == 0:
        raise ValueError()
    elif not os.path.isfile(path):
        raise FileNotFoundError()

    with open(path, 'r', encoding="utf8") as f:
        config = []
        for l in f.readlines():
            k, v = l.strip().split('=')
            config.append((k, int(v)))
        return CapsNetParam(**dict(config))


def make_param(image_width: int = 28,image_height: int = 28,image_channel: int = 1,conv1_filter: int = 32,conv1_kernel: int = 5,conv1_stride: int = 1,conv2_filter: int = 64,conv2_kernel: int = 3,conv2_stride: int = 1,conv3_filter: int = 64,conv3_kernel: int = 3,conv3_stride: int = 1,conv4_filter: int = 128,conv4_kernel: int = 3,conv4_stride: int = 2,dconv_kernel: int = 9,dconv_stride: int = 1,num_primary_caps: int = 16,dim_primary_caps: int = 8,num_digit_caps: int = 10,dim_digit_caps: int = 16) -> CapsNetParam:
    return CapsNetParam(image_width,image_height,image_channel,conv1_filter,conv1_kernel,conv1_stride,conv2_filter,conv2_kernel,conv2_stride,conv3_filter,conv3_kernel,conv3_stride,conv4_filter,conv4_kernel,conv4_stride,dconv_kernel,dconv_stride,num_primary_caps,dim_primary_caps,num_digit_caps,dim_digit_caps,
    )

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir",
                    None,
                    "Directory to save training results.",
                    required=True)
flags.DEFINE_integer("num_epochs",
                     3,
                     "Number of epochs.",
                     lower_bound=0,
                     upper_bound=100)
flags.DEFINE_float("validation_split",
                   0.2,
                   "Ratio for a validation dataset from training dataset.",
                   lower_bound=0.0,
                   upper_bound=0.5)
flags.DEFINE_boolean("show_score", False,
                     "Flag for scoring the trained model.")
flags.DEFINE_boolean("show_summary", False,
                     "Flag for displaying the model summary.")
flags.DEFINE_boolean("scale_mnist", False,
                     "Flag for scaling the MNIST dataset.")
flags.DEFINE_boolean("plot_logs", False, "Flag for plotting the saved logs.")


'''In this function, we load the MNIST dataset and return the training and testing dataset.'''
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


def _plot_training_logs(checkpoint_dir: str, dpi: int = 300) -> None:
    with open(f"{checkpoint_dir}/train_log.csv", mode='r') as csvfile:
        logs = np.array(
            [line.strip().split(',') for line in csvfile.readlines()])
        logs = logs[1:, :].astype(np.float)  # [1:,:]: remove header

        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(logs[:, 1], label="Training accuracy")
        plt.plot(logs[:, 3], label="Validation accuracy")
        plt.legend()
        plt.savefig(f"{checkpoint_dir}/accuracy.png", dpi=dpi)

        plt.clf()

        plt.title("Margin loss")
        plt.xlabel("Epoch")
        plt.ylabel("Margin loss")
        plt.plot(logs[:, 2], label="Training loss")
        plt.plot(logs[:, 4], label="Validation loss")
        plt.legend()
        plt.savefig(f"{checkpoint_dir}/loss.png", dpi=dpi)

'''In this function, we define the main function for training and testing the Efficient CapsNet model.
We first define the parameters for the model and load the MNIST dataset.
Then, we define the checkpoint directory and the number of epochs for training.
If the checkpoint directory already exists, we load the latest checkpoint and continue training from there.
We also define the callbacks for saving the training logs and the checkpoints.
Finally, we train the model and save the training logs and the checkpoints.
'''
def main(_) -> None:
    param = make_param()
    model = efficient_capsnet.make_model(param)
    mnist_train, mnist_test = _get_mnist_dataset(param.num_digit_caps,
                                                 FLAGS.scale_mnist)
    X_train, y_train = mnist_train
    X_test, y_test = mnist_test

    checkpoint_dir = FLAGS.checkpoint_dir
    initial_epoch = 0
    num_epochs = FLAGS.num_epochs

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
              callbacks=[csv_logger, model_saver],
              batch_size= 16)
    model.save(f"{checkpoint_dir}/model")
    param.save_config(f"{checkpoint_dir}/config.txt")

    if FLAGS.show_score is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            loss, score = model.evaluate(X_test, y_test)
            print(f"Test loss: {loss: .4f}")
            print(f"Test score: {score: .4f}")

    if FLAGS.show_summary is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            model.summary()

    if FLAGS.plot_logs is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            _plot_training_logs(checkpoint_dir, dpi=300)


if __name__ == "__main__":
    app.run(main)