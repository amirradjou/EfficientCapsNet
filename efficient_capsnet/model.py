import tensorflow as tf


class Squash(tf.keras.layers.Layer):
    def __init__(self, eps: float = 1e-7, name: str = "squash") -> None:
        super(Squash, self).__init__(name=name)
        self.eps = eps

    def call(self, input_vector: tf.Tensor) -> tf.Tensor:
        norm = tf.norm(input_vector, axis=-1, keepdims=True)
        coef = 1 - 1 / tf.exp(norm)
        unit = input_vector / (norm + self.eps)
        return coef * unit

    def compute_output_shape(self,
                             input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape

