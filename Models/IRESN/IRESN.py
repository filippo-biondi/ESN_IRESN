import tensorflow as tf
from tensorflow import keras
from keras.layers import Masking, RNN, Lambda
import keras.backend as BK
from Models.classifiers import RidgeClassifier, LassoClassifier
from keras.models import Model

from Models.IRESN.ReservoirCell import ReservoirCell


def add_bias(states):
    # add bias
    states = tf.concat([states, tf.ones((states.shape[0], 1))], axis=-1)
    return states


class IRESN(Model):
    def __init__(self,
                 units,
                 sub_reservoirs,
                 spectral_radius=0.999,
                 leaky=0.1,
                 input_scaling=1.,
                 bias_scaling=0.1,
                 reservoir_activation=tf.keras.activations.tanh,
                 reg=None,
                 lasso=False,
                 **kwargs
                 ):

        super().__init__(**kwargs)
        self.lasso = lasso

        cell = ReservoirCell(units,
                             leaky=leaky,
                             sub_reservoirs=sub_reservoirs,
                             activation=reservoir_activation,
                             spectral_radius=spectral_radius,
                             input_scaling=input_scaling,
                             bias_scaling=bias_scaling)

        self.reservoir = keras.Sequential([
            Masking(mask_value=-2),
            RNN(cell=cell, return_sequences=True),
            Lambda(lambda x: BK.mean(x, axis=1))])
        self._reg = reg

    def computeStates(self, inputs, *args, **kwargs):
        return self.reservoir(inputs, *args, **kwargs)

    def computeOutput(self, states):
        states = add_bias(states)
        return self.readout.predict(states)

    def call(self, inputs, *args, **kwargs):
        states = self.computeStates(inputs, *args, **kwargs)
        return self.computeOutput(states)

    def predict_cont(self, inputs, *args, **kwargs):
        states = self.computeStates(inputs, *args, **kwargs)
        states = add_bias(states)
        return self.readout.predict_cont(states)

    def trainReadout(self, states, y, **kwargs):
        reg = self._reg
        if reg is None or "reg" in kwargs:
            reg = kwargs["reg"]
        if self.lasso:
            self.readout = LassoClassifier(alpha=reg)
        else:
            self.readout = RidgeClassifier(alpha=reg)
        states = add_bias(states)
        self.readout.fit(states, y)
        return self.readout.score(states, y)

    def fit(self, x, y, **kwargs):
        results = {}
        states = self.computeStates(x)
        results["train_accuracy"] = self.trainReadout(states, y, **kwargs)
        return results

    def evaluate(self, x, y, **kwargs):
        states = self.computeStates(x)
        return self.evaluateReadout(states, y)

    def evaluateReadout(self, states, y):
        states = add_bias(states)
        return self.readout.score(states, y)

    def evaluate_precise(self, x, y):
        tp, fp, tn, fn = 0, 0, 0, 0

        pred = self.call(x)
        total = len(pred)

        for i in range(total):
            if y[i] == 1 and pred[i] == 1:
                tp += 1
            elif y[i] == 0 and pred[i] == 0:
                tn += 1
            elif y[i] == 0 and pred[i] == 1:
                fp += 1
            else:
                fn += 1

        accuracy = (tp + tn) / total
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return accuracy, sensitivity, specificity, {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
