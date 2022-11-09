import tensorflow as tf
import keras
from keras import Model
from keras.layers import Lambda, Masking, RNN, Dense
import keras.backend as BK

from Models.TrainableESN.ReservoirCell import ReservoirCell


class TrainableESN(Model):
    def __init__(self,
                 units,
                 output_units=1,
                 spectral_radius=0.999,
                 input_scaling=1.,
                 bias_scaling=0.1,
                 leaky=0.1,
                 reservoir_activation=tf.keras.activations.tanh,
                 readout_activation=tf.keras.activations.sigmoid,
                 mask_value=-2,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.use_bias = bias_scaling is not None

        cell = ReservoirCell(units=units, leaky=leaky,
                             activation=reservoir_activation,
                             use_bias=self.use_bias,
                             input_scaling=input_scaling,
                             bias_scaling=bias_scaling,
                             spectral_radius=spectral_radius)

        self.reservoir = keras.Sequential([
            Masking(mask_value=mask_value),
            RNN(cell=cell, return_sequences=True),
            Lambda(lambda x: BK.mean(x, axis=1))])

        self.readout = keras.Sequential([
            Dense(output_units, activation=readout_activation, name="readout")
        ])

    def computeStates(self, x):
        return self.reservoir(x)

    def computeOutput(self, states):
        return self.readout(states)

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

    def call(self, x, **kwargs):
        return self.computeOutput(self.computeStates(x))
