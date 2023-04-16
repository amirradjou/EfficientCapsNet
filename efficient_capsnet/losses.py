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
    
        def get_config(self) -> dict:
            return {
            "present_max": self.present_max,
            "absent_min": self.absent_min,
            "absent_scale": self.absent_scale
        }
    def from_config(config):
        return MarginLoss(**config)
    def __repr__(self):
        return f"MarginLoss(present_max={self.present_max}, absent_min={self.absent_min}, absent_scale={self.absent_scale})"
    def __str__(self):
        return self.__repr__()
    def __eq__(self, other):

        if not isinstance(other, MarginLoss):
            return False
        return (self.present_max == other.present_max and
                self.absent_min == other.absent_min and
                self.absent_scale == other.absent_scale)
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash((self.present_max, self.absent_min, self.absent_scale))
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __reduce__(self):
        return (self.__class__, (self.present_max, self.absent_min, self.absent_scale))
    def __reduce_ex__(self, protocol):
        return self.__reduce__()
    def __sizeof__(self):
        return object.__sizeof__(self)
    def __dir__(self):
        return object.__dir__(self)
    def __format__(self, format_spec):
        return object.__format__(self, format_spec)
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)
    def __setattr__(self, name, value):
        return object.__setattr__(self, name, value)
    def __delattr__(self, name):
        return object.__delattr__(self, name)
    
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
    
