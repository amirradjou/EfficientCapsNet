from efficient_capsnet.layers import DigitCap
from efficient_capsnet.layers import FeatureMap
from efficient_capsnet.layers import PrimaryCap
from efficient_capsnet.losses import MarginLoss

import tensorflow as tf
from typing import List
from typing import Union

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
        with open(path, 'w', encoding='utf8') as f:
            for k, v in self.get_config().items():
                f.writelines(f"{k}={v}\n")


def load_config(path: str) -> CapsNetParam:
    with open(path, 'r', encoding="utf8") as f:
        config = []
        for l in f.readlines():
            k, v = l.strip().split('=')
            config.append((k, int(v)))
        return CapsNetParam(**dict(config))


def make_param(image_width: int = 28,image_height: int = 28,image_channel: int = 1,conv1_filter: int = 32,conv1_kernel: int = 5,conv1_stride: int = 1,conv2_filter: int = 64,conv2_kernel: int = 3,conv2_stride: int = 1,conv3_filter: int = 64,conv3_kernel: int = 3,conv3_stride: int = 1,conv4_filter: int = 128,conv4_kernel: int = 3,conv4_stride: int = 2,dconv_kernel: int = 9,dconv_stride: int = 1,num_primary_caps: int = 16,dim_primary_caps: int = 8,num_digit_caps: int = 10,dim_digit_caps: int = 16) -> CapsNetParam:
    return CapsNetParam(image_width,image_height,image_channel,conv1_filter,conv1_kernel,conv1_stride,conv2_filter,conv2_kernel,conv2_stride,conv3_filter,conv3_kernel,conv3_stride,conv4_filter,conv4_kernel,conv4_stride,dconv_kernel,dconv_stride,num_primary_caps,dim_primary_caps,num_digit_caps,dim_digit_caps,
    )
def make_param_from_config(path: str) -> CapsNetParam:
    return load_config(path)


'''make_model is used to build the model. We use the same architecture as mentioned in the paper: Efficient CapsNet.
In building the model, for the first layer, we use the input_shape parameter to specify the input shape.
for the rest of the layers, we use the output_shape of the previous layer as the input_shape.'''
def make_model(
    param: CapsNetParam,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    loss: tf.keras.losses.Loss = MarginLoss(),
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy"]
) -> tf.keras.Model:
    input_images = tf.keras.layers.Input(
        shape=[param.input_height, param.input_width, param.input_channel],
        name="input_images")
    feature_maps = FeatureMap(param, name="feature_maps")(input_images)
    primary_caps = PrimaryCap(param, name="primary_caps")(feature_maps)
    digit_caps = DigitCap(param, name="digit_caps")(primary_caps)
    digit_probs = tf.keras.layers.Lambda(lambda x: tf.norm(x, axis=-1),
                                         name="digit_probs")(digit_caps)

    model = tf.keras.Model(inputs=input_images,
                           outputs=digit_probs,
                           name="Efficient-CapsNet")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def make_model_from_config(
    path: str,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    loss: tf.keras.losses.Loss = MarginLoss(),
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy"]
) -> tf.keras.Model:
    param = load_config(path)
    return make_model(param, optimizer, loss, metrics)
def make_model_from_param(
    param: CapsNetParam,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    loss: tf.keras.losses.Loss = MarginLoss(),
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy"]
) -> tf.keras.Model:
    return make_model(param, optimizer, loss, metrics)
def make_model_from_config_file(
    path: str,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    loss: tf.keras.losses.Loss = MarginLoss(),
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy"]
) -> tf.keras.Model:
    param = load_config(path)
    return make_model(param, optimizer, loss, metrics)
def make_model_from_param_file(
    path: str,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    loss: tf.keras.losses.Loss = MarginLoss(),
    metrics: List[Union[str, tf.keras.metrics.Metric]] = ["accuracy"]
) -> tf.keras.Model:
    param = load_config(path)
    return make_model(param, optimizer, loss, metrics)
