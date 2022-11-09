import tensorflow as tf
import numpy as np
from tensorflow import keras


class ReservoirCell(keras.layers.Layer):
    # builds the reservoir system for an input-routed ESN:
    # the system contains a number of sub-reservoir systems, each fed by 1 specific input dimension
    # inter-reservoir connections are modulated by specific hyper-parameters

    def __init__(self, units,
                 sub_reservoirs,
                 input_scaling=1.0,
                 bias_scaling=0.1,
                 spectral_radius=0.999,
                 leaky=0.1,
                 activation=tf.nn.tanh,
                 **kwargs):

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky  # leaking rate
        self.activation = activation
        self.num_reservoirs = sub_reservoirs
        if isinstance(units, list):
            self.units = sum(units)
            self.num_units_reservoir = units
        else:
            self.units = units
            self.num_units_reservoir = [int(self.units / self.num_reservoirs) for _ in range(self.num_reservoirs)]
        self.state_size = self.units
        if not isinstance(input_scaling, list):
            self.input_scaling = [input_scaling for _ in range(self.num_reservoirs)]
        if not isinstance(bias_scaling, list):
            self.bias_scaling = [bias_scaling for _ in range(self.num_reservoirs)]
        if not isinstance(spectral_radius, list):
            self.spectral_radius = [spectral_radius for _ in range(self.num_reservoirs)]
        super().__init__(**kwargs)

        # as for standard ESN, when building the recurrent weight matrix
        # uses circular law to determine the values of the recurrent weight matrix
        # rif. paper
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.

        W = np.zeros(shape=(self.units, self.units), dtype=np.float32)

        units_used = 0
        # recurrent kernel
        for i in range(self.num_reservoirs):
            value = (self.spectral_radius[i] / np.sqrt(self.num_units_reservoir[i])) * (6 / np.sqrt(12))
            W[units_used:units_used + self.num_units_reservoir[i],
            units_used:units_used + self.num_units_reservoir[i]] = tf.random.uniform(
                shape=(self.num_units_reservoir[i], self.num_units_reservoir[i]), minval=-value, maxval=value)
            units_used += self.num_units_reservoir[i]
        self.recurrent_kernel = W

        # input weight matrix
        Win = np.zeros(shape=(self.num_reservoirs, self.units), dtype=np.float32)

        units_used = 0
        for i in range(self.num_reservoirs):
            Win[i, units_used:units_used + self.num_units_reservoir[i]] = tf.random.uniform(
                shape=(1, self.num_units_reservoir[i]), minval=-self.input_scaling[i], maxval=self.input_scaling[i])
            units_used += self.num_units_reservoir[i]
        self.kernel = Win

        # bias
        b = np.zeros(shape=(self.units,), dtype=np.float32)

        units_used = 0
        for i in range(self.num_reservoirs):
            b[units_used:units_used + self.num_units_reservoir[i]] = tf.random.uniform(
                shape=(self.num_units_reservoir[i],),
                minval=-self.bias_scaling[i],
                maxval=self.bias_scaling[i])
            units_used += self.num_units_reservoir[i]
        self.bias = b
        self.built = True


    def call(self, inputs, states):
        in_matrix = tf.concat([inputs, states[0]], axis=1, name="in->state")
        weights_matrix = tf.concat([self.kernel, self.recurrent_kernel], axis=0, name="ker->rec")

        output = tf.matmul(in_matrix, weights_matrix, name="bigmul")

        output = output + self.bias
        output = self.activation(output)
        output = (1 - self.leaky) * states[0] + self.leaky * output

        return output, [output]

