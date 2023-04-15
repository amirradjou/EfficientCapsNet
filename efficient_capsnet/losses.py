import tensorflow as tf

#Margin Loss
''' We use the margin loss function from the paper: Dynamic Routing Between Capsules
 which was also mentioned in Efficient CapsNet paper.'''
class MarginLoss(tf.keras.losses.Loss):
    def __init__(self,
                 present_max: float = 0.9,
                 absent_min: float = 0.1,
                 absent_scale: float = 0.5) -> None:
        super(MarginLoss, self).__init__(name="MarginLoss")
        self.present_max = present_max
        self.absent_min = absent_min
        self.absent_scale = absent_scale

    ''' The margin loss function is defined as: 
    L = T_c * max(0, m_plus - ||v_c||)^2 + lambda * (1 - T_c) * max(0, ||v_c|| - m_minus)^2 where 
    T_c is the target for digit c, m_plus is the desired'''
    def call(self, labels: tf.Tensor, digit_probs: tf.Tensor) -> tf.Tensor:
        assert labels.shape is not digit_probs.shape
        zeros = tf.zeros_like(labels, dtype=tf.float32)
        present_losses = labels * tf.square(
            tf.maximum(zeros, self.present_max - digit_probs))
        absent_losses = (1 - labels) * tf.square(
            tf.maximum(zeros, digit_probs - self.absent_min))
        losses = present_losses + self.absent_scale * absent_losses
        return tf.reduce_sum(losses, axis=-1, name="total_loss")