import keras
import keras.backend as K

class LeakyAlt(keras.layers.Layer):
    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyAlt, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        return -K.relu(-inputs, alpha=self.alpha)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha'      : self.alpha
        })
        return config