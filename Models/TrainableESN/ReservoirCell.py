import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer

from Models import initializers


class ReservoirCell(Layer):
    def __init__(
            self,
            units: int,
            leaky: float = 0.1,
            activation=tf.keras.activations.tanh,
            connectivity=1.0,
            input_scaling=1.0,
            bias_scaling=1.0,
            spectral_radius=1.0,
            use_bias: bool = True,
            **kwargs,
    ):

        super().__init__(name="reservoir", **kwargs)

        self.units = units
        self.activation = activation

        self.leaky = leaky
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius

        self._state_size = units
        self._output_size = units

        self.use_bias = use_bias

        self.kernel_initializer = initializers.Kernel(
            initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1))

        if connectivity == 1.0:
            self.recurrent_initializer = initializers.FullConnected(1)
        else:
            self.recurrent_initializer = initializers.RecurrentKernel(connectivity, 1)

        if self.use_bias:
            self.bias_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        else:
            self.bias_initializer = None

        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.built = False

    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(tf.TensorShape(inputs_shape)[-1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]. Shape received is %s"
                % inputs_shape
            )

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self.units],
            initializer=self.kernel_initializer,
            trainable=False,
            dtype=self.dtype)

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.units, self.units],
            initializer=self.recurrent_initializer,
            trainable=False,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                trainable=False,
                dtype=self.dtype)

        self.input_scaling = self.add_weight(name="input_scaling", shape=(1,), dtype=tf.float32,
                                             initializer=keras.initializers.Constant(self.input_scaling),
                                             trainable=True)
        self.bias_scaling = self.add_weight(name="bias_scaling", shape=(1,), dtype=tf.float32,
                                            initializer=keras.initializers.Constant(self.bias_scaling), trainable=True)
        self.spectral_radius = self.add_weight(name="spectral_radius", shape=(1,), dtype=tf.float32,
                                               initializer=keras.initializers.Constant(self.spectral_radius),
                                               trainable=True)
        self.leaky = self.add_weight(name="leaky", shape=(1,), dtype=tf.float32,
                                     initializer=keras.initializers.Constant(self.leaky), trainable=False)

        self.built = True

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    @tf.function
    def call(self, inputs, states):
        in_matrix = tf.concat([inputs, states[0]], axis=1, name="in->state")
        weights_matrix = tf.concat([self.kernel * self.input_scaling, self.recurrent_kernel * self.spectral_radius],
                                   axis=0, name="ker->rec")

        output = tf.matmul(in_matrix, weights_matrix, name="bigmul")
        if self.use_bias:
            output = output + self.bias * self.bias_scaling
        output = self.activation(output)
        output = (1 - self.leaky) * states[0] + self.leaky * output

        return output, [output]