import tensorflow as tf
from keras.layers import Masking, RNN, Lambda
from tensorflow import keras
import keras.backend as BK

from Models.TrainableIRESN.ReservoirCell import ReservoirCell

class TrainableIRESN(keras.Model):
    # Implements an Echo State Network model for time-series classification problems
    #
    # The architecture comprises a recurrent layer with ReservoirCell,
    # followed by a trainable dense readout layer for classification

    def __init__(self, units,
                 sub_reservoirs,
                 output_units=1,
                 output_activation=tf.nn.sigmoid,
                 input_scaling=1.,
                 inter_scaling=None,
                 bias_scaling=0.1,
                 spectral_radius=0.999,
                 leaky=0.1,
                 activation=tf.nn.tanh,
                 mask_value=-2.,
                 **kwargs):
        super().__init__(**kwargs)

        cell = ReservoirCell(units=units,
                             sub_reservoirs=sub_reservoirs,
                             input_scaling=input_scaling,
                             inter_scaling=inter_scaling,
                             bias_scaling=bias_scaling,
                             spectral_radius=spectral_radius,
                             leaky=leaky,
                             activation=activation)

        self.reservoir = keras.Sequential([
            Masking(mask_value=mask_value),
            RNN(cell=cell, return_sequences=True),
            Lambda(lambda x: BK.mean(x, axis=1))
        ])

        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation)
        ])

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.readout.compile(*args, **kwargs)

    def computeStates(self, inputs):
        return self.reservoir(inputs)

    def computeOutput(self, states):
        return self.readout(states)

    def call(self, x):
        return self.computeOutput(self.computeStates(x))

    def train_step(self, data):
        x, y = data
        y = tf.cast(y, dtype=tf.float32)
        with tf.GradientTape() as tape:
            states = self.reservoir(x)
            y_pred = self.readout(states)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, dtype=tf.float32)
        states = self.reservoir(x)
        y_pred = self.readout(states)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def evaluate_precise(self, x, y):
        tp, fp, tn, fn = 0, 0, 0, 0

        pred = self.call(x)
        total = len(pred)

        for i in range(total):
            if y[i] == 1 and pred[i] >= 0.5:
                tp += 1
            elif y[i] == 0 and pred[i] < 0.5:
                tn += 1
            elif y[i] == 0 and pred[i] >= 0.5:
                fp += 1
            else:
                fn += 1

        accuracy = (tp + tn) / total
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return accuracy, sensitivity, specificity, {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

