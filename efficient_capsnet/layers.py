import tensorflow as tf


import os

class CapsNetParam(object):
    __slots__ = ["input_width","input_height","input_channel","conv1_filter","conv1_kernel","conv1_stride","conv2_filter","conv2_kernel","conv2_stride","conv3_filter","conv3_kernel","conv3_stride","conv4_filter","conv4_kernel","conv4_stride","dconv_filter","dconv_kernel","dconv_stride","num_primary_caps","dim_primary_caps","num_digit_caps","dim_digit_caps",]
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
    def __repr__(self) -> str:
        return str(self.get_config())
    def __str__(self) -> str:
        return str(self.get_config())
    def __eq__(self, other) -> bool:
        return self.get_config() == other.get_config()
    def __ne__(self, other) -> bool:
        return self.get_config() != other.get_config()
    def __hash__(self) -> int:
        return hash(self.get_config())

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

'''Squash Class is used to squash the output of the capsule layer.'''
class Squash(tf.keras.layers.Layer):
    def __init__(self, eps: float = 1e-7, name: str = "squash") -> None:
        super(Squash, self).__init__(name=name)
        self.eps = eps


    def call(self, input_vector: tf.Tensor) -> tf.Tensor:
        norm = tf.norm(input_vector, axis=-1, keepdims=True)
        return (1 - 1 / tf.exp(norm)) * (input_vector / (norm + self.eps))

    def compute_output_shape(self,input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape
    
    def get_config(self) -> dict:
        return {"eps": self.eps}
    


'''FeatureMap Class is used to extract the feature maps from the input images. 
We use the same architecture as mentioned in the paper: Efficient CapsNet. In building the model,
for the first layer, we use the input_shape parameter to specify the input shape.
for the rest of the layers, we use the output_shape of the previous layer as the input_shape.'''
class FeatureMap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam, name: str = "FeatureMap") -> None:
        super(FeatureMap, self).__init__(name=name)
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv1 = tf.keras.layers.Conv2D(input_shape=input_shape[1:],filters=self.param.conv1_filter,kernel_size=self.param.conv1_kernel,strides=self.param.conv1_stride,activation=tf.keras.activations.relu,name="feature_map_conv1")
        self.norm1 = tf.keras.layers.BatchNormalization(name="feature_map_norm1")
        self.conv2 = tf.keras.layers.Conv2D(filters=self.param.conv2_filter,kernel_size=self.param.conv2_kernel,strides=self.param.conv2_stride,activation=tf.keras.activations.relu,name="feature_map_conv2")
        self.norm2 = tf.keras.layers.BatchNormalization(name="feature_map_norm2")
        self.conv3 = tf.keras.layers.Conv2D(filters=self.param.conv3_filter,kernel_size=self.param.conv3_kernel,strides=self.param.conv3_stride,activation=tf.keras.activations.relu,name="feature_map_conv3")
        self.norm3 = tf.keras.layers.BatchNormalization(name="feature_map_norm3")
        self.conv4 = tf.keras.layers.Conv2D(filters=self.param.conv4_filter,kernel_size=self.param.conv4_kernel,strides=self.param.conv4_stride,activation=tf.keras.activations.relu,name="feature_map_conv4")
        self.norm4 = tf.keras.layers.BatchNormalization(name="feature_map_norm4")
        self.built = True

    def call(self, input_images: tf.Tensor) -> tf.Tensor:
        feature_maps = self.norm1(self.conv1(input_images))
        feature_maps = self.norm2(self.conv2(feature_maps))
        feature_maps = self.norm3(self.conv3(feature_maps))
        return self.norm4(self.conv4(feature_maps))

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        output_shape = self.conv1.compute_output_shape(input_shape)
        output_shape = self.conv2.compute_output_shape(output_shape)
        output_shape = self.conv3.compute_output_shape(output_shape)
        output_shape = self.conv4.compute_output_shape(output_shape)
        return output_shape

'''PrimaryCap Class is used to extract the primary capsules from the feature maps.
In building the model, we use the output_shape of the previous layer as the input_shape.
Refer to the paper: Efficient CapsNet for more details.'''
class PrimaryCap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam, name: str = "PrimaryCap") -> None:
        super(PrimaryCap, self).__init__(name=name)
        self.param = param

    def build(self, input_shape: tf.TensorShape) -> None:
        self.dconv = tf.keras.layers.Conv2D(input_shape=input_shape[1:],filters=self.param.dconv_filter,kernel_size=self.param.dconv_kernel,strides=self.param.dconv_stride,groups=self.param.dconv_filter,activation=tf.keras.activations.relu,name="primary_cap_dconv")
        self.reshape = tf.keras.layers.Reshape(target_shape=[-1, self.param.dim_primary_caps],name="primary_cap_reshape")
        self.squash = Squash(name="primary_cap_squash")
        self.built = True

    def call(self, feature_maps: tf.Tensor) -> tf.Tensor:
        dconv_outputs = self.dconv(feature_maps)
        return self.squash(self.reshape(dconv_outputs))

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        output_shape = self.dconv.compute_output_shape(input_shape)
        output_shape = self.reshape.compute_output_shape(output_shape)
        return output_shape

'''DigitCap Class is used to extract the digit capsules from the primary capsules. 
For each digit capsule, we use the primary capsules as the input.
In building the model, we use the output_shape of the previous layer as the input_shape. 
Refer to the paper: Efficient CapsNet for more details.'''
class DigitCap(tf.keras.layers.Layer):
    def __init__(self, param: CapsNetParam, name="DigitCap") -> None:
        super(DigitCap, self).__init__(name=name)
        self.param = param
        self.attention_coef = 1 / tf.sqrt(tf.cast(self.param.dim_primary_caps, dtype=tf.float32))

    def build(self, input_shape: tf.TensorShape) -> None:
        self.num_primary_caps = input_shape[1]
        self.dim_primary_caps = input_shape[2]
        self.W = self.add_weight(name="digit_caps_transform_tensor",shape=(self.param.num_digit_caps, self.num_primary_caps,self.param.dim_digit_caps, self.dim_primary_caps),dtype=tf.float32,initializer="glorot_uniform",trainable=True)
        self.B = self.add_weight(name="digit_caps_log_priors",shape=[self.param.num_digit_caps, 1, self.num_primary_caps],dtype=tf.float32,initializer="glorot_uniform",trainable=True)
        self.squash = Squash(name="digit_cap_squash")
        self.built = True

    '''
    U is the input of the digit capsule. U.shape: [None, num_digit_caps, num_primary_caps, dim_primary_caps, 1]
    U_hat is the prediction of the digit capsule.  U_hat.shape: [None, num_digit_caps, num_primary_caps, dim_digit_caps]
    A is the attention matrix. A.shape: [None, num_digit_caps, num_primary_caps, num_primary_caps]
    C is the coupling coefficient matrix. C.shape: [None, num_digit_caps, 1, num_primary_caps]
    S is the output of the digit capsule. S.shape: [None, num_digit_caps, dim_digit_caps]
    '''
    def call(self, primary_caps: tf.Tensor) -> tf.Tensor:
        U = tf.expand_dims(tf.tile(tf.expand_dims(primary_caps, axis=1),[1, self.param.num_digit_caps, 1, 1]),axis=-1,name="digit_cap_inputs")
        U_hat = tf.squeeze(tf.map_fn(lambda u_i: tf.matmul(self.W, u_i), U),axis=-1,name="digit_cap_predictions")
        A = self.attention_coef * tf.matmul(U_hat, U_hat, transpose_b=True, name="digit_cap_attentions")
        C = tf.nn.softmax(tf.reduce_sum(A, axis=-2, keepdims=True),axis=-2,name="digit_cap_coupling_coefficients")
        S = tf.squeeze(tf.matmul(C + self.B, U_hat), axis=-2)
        return self.squash(S)
    
    def compute_output_shape(self,input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([input_shape[0],self.param.num_digit_caps,self.param.dim_digit_caps,])