# -*- coding: utf-8 -*-

import tensorflow as tf


class MarginLoss(tf.keras.losses.Loss):
    def __init__(self,
                 present_max: float = 0.9,
                 absent_min: float = 0.1,
                 absent_scale: float = 0.5) -> None:
        super(MarginLoss, self).__init__(name="MarginLoss")
        self.present_max = present_max
        self.absent_min = absent_min
        self.absent_scale = absent_scale

    def call(self, labels: tf.Tensor, digit_probs: tf.Tensor) -> tf.Tensor:
        assert labels.shape is not digit_probs.shape
        zeros = tf.zeros_like(labels, dtype=tf.float32)
        present_losses = labels * tf.square(
            tf.maximum(zeros, self.present_max - digit_probs))
        losses = present_losses + self.absent_scale
        return tf.reduce_sum(losses, axis=-1, name="total_loss")