import tensorflow as tf
import numpy as np
from keras.layers import Layer


class ReservoirCell(Layer):
    # builds a reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, units,
                 input_scaling=1.0,
                 bias_scaling=0.1,
                 spectral_radius=0.99,
                 leaky=0.1,
                 activation=tf.nn.tanh,
                 **kwargs):
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky  # leaking rate
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):

        # build the input weight matrix
        self.kernel = tf.random.uniform(shape=(input_shape[-1], self.units), minval=-self.input_scaling,
                                        maxval=self.input_scaling)

        # build the recurrent weight matrix
        # uses circular law to determine the values of the recurrent weight matrix
        # rif. paper
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))
        W = tf.random.uniform(shape=(self.units, self.units), minval=-value, maxval=value)
        self.recurrent_kernel = W

        # initialize the bias
        self.bias = tf.random.uniform(shape=(self.units,), minval=-self.bias_scaling, maxval=self.bias_scaling)

        self.built = True

    @tf.function
    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(inputs, self.kernel)

        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = prev_output * (1 - self.leaky) + self.activation(input_part + self.bias + state_part) * self.leaky
        else:
            output = prev_output * (1 - self.leaky) + (input_part + self.bias + state_part) * self.leaky

        return output, [output]