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
                 inter_scaling=None,
                 spectral_radius=0.999,
                 leaky=0.1,
                 activation=tf.nn.tanh,
                 **kwargs):
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.inter_scaling = inter_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.activation = activation
        self.num_reservoirs = sub_reservoirs
        self.num_units_reservoir = int(self.units / self.num_reservoirs)
        super().__init__(**kwargs)

        # as for standard ESN, when building the recurrent weight matrix
        # uses circular law to determine the values of the recurrent weight matrix
        # rif. paper
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.

        W = np.zeros(shape=(self.units, self.units), dtype=np.float32)

        # recurrent kernel
        for i in range(self.num_reservoirs):
            value = (1.0 / np.sqrt(self.num_units_reservoir)) * (6 / np.sqrt(12))
            W[self.num_units_reservoir * i:self.num_units_reservoir * (i + 1),
            self.num_units_reservoir * i:self.num_units_reservoir * (i + 1)] = tf.random.uniform(
                shape=(self.num_units_reservoir, self.num_units_reservoir), minval=-value, maxval=value)
            for j in range(self.num_reservoirs):
                if i != j and self.inter_scaling is not None:
                    W[self.num_units_reservoir * j:self.num_units_reservoir * (j + 1),
                    self.num_units_reservoir * i:self.num_units_reservoir * (i + 1)] = tf.random.uniform(
                        shape=(self.num_units_reservoir, self.num_units_reservoir), minval=-1.0,
                        maxval=1.0)
        self.recurrent_kernel = W

        # input weight matrix
        Win = np.zeros(shape=(self.num_reservoirs, self.units), dtype=np.float32)
        for i in range(self.num_reservoirs):
            Win[i, self.num_units_reservoir * i:self.num_units_reservoir * (i + 1)] = tf.random.uniform(
                shape=(1, self.num_units_reservoir), minval=-1.0, maxval=1.0)
        self.kernel = Win

        # bias
        b = np.zeros(shape=(self.units,), dtype=np.float32)
        for i in range(self.num_reservoirs):
            b[self.num_units_reservoir * i:self.num_units_reservoir * (i + 1)] = tf.random.uniform(
                shape=(self.num_units_reservoir,),
                minval=-1.0,
                maxval=1.0)
        self.bias = b

        em = np.zeros(shape=(self.units, self.num_reservoirs), dtype=np.float32)
        for i in range(self.num_reservoirs):
            em[i * self.num_units_reservoir:(i + 1) * self.num_units_reservoir, i] = tf.ones(
                shape=(self.num_units_reservoir,))
        self.em = em

        self.input_scaling = self.add_weight(name="input_scaling", shape=(self.num_reservoirs,), dtype=tf.float32,
                                             initializer=keras.initializers.Constant(self.input_scaling),
                                             trainable=True)
        self.bias_scaling = self.add_weight(name="bias_scaling", shape=(self.num_reservoirs, 1), dtype=tf.float32,
                                            initializer=keras.initializers.Constant(self.bias_scaling), trainable=True)
        if self.inter_scaling is not None:
            self.inter_scaling = self.add_weight(name="inter_scaling", shape=(self.num_reservoirs,), dtype=tf.float32,
                                                 initializer=keras.initializers.Constant(self.inter_scaling),
                                                 trainable=True)
        self.spectral_radius = self.add_weight(name="spectral_radius", shape=(self.num_reservoirs,), dtype=tf.float32,
                                               initializer=keras.initializers.Constant(self.spectral_radius),
                                               trainable=False)
        self.leaky = self.add_weight(name="leaky", shape=(1,), dtype=tf.float32,
                                     initializer=keras.initializers.Constant(self.leaky), trainable=False)

        self.built = True

    @tf.function
    def call(self, inputs, states):
        prev_output = states[0]

        input_part = tf.matmul(tf.math.multiply(inputs, self.input_scaling), self.kernel)

        if self.inter_scaling is None:
            rec_scaling = tf.linalg.matvec(self.em, self.spectral_radius)
        else:
            t = tf.stack([self.spectral_radius] + [self.inter_scaling for _ in range(self.num_reservoirs - 1)], axis=0)
            m = []
            for i in range(self.num_reservoirs):
                c = t[:, i]
                c = tf.roll(c, shift=i, axis=0)
                m.append(c)
            m = tf.stack(m, axis=1)
            m = tf.matmul(self.em, m)
            rec_scaling = tf.matmul(m, tf.transpose(self.em))

        state_part = tf.matmul(prev_output, tf.math.multiply(self.recurrent_kernel, rec_scaling))

        ext_bias = tf.reshape(tf.matmul(self.em, self.bias_scaling), shape=(self.units,))
        if self.activation is not None:
            output = prev_output * (1 - self.leaky) + self.activation(
                input_part + tf.math.multiply(self.bias, ext_bias) + state_part) * self.leaky
        else:
            output = prev_output * (1 - self.leaky) + (input_part + self.bias + state_part) * self.leaky

        return output, [output]

