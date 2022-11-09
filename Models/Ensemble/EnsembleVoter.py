from keras import Model
import numpy as np

class EnsembleVoter(Model):

    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models

    def fit(self, x=None, y=None, *args, **kwargs):
        return

    def evaluate(self, x=None, y=None, *args, **kwargs):
        return np.mean(y == self.call(x))

    def evaluate_precise(self, x, y):
        tp, fp, tn, fn = 0, 0, 0, 0

        preds = self.call(x)
        total = len(preds)

        for i, pred in enumerate(preds):
            if y[i] == 1 and pred == 1:
                tp += 1
            elif y[i] == 0 and pred == 0:
                tn += 1
            elif y[i] == 0 and pred == 1:
                fp += 1
            else:
                fn += 1

        accuracy = (tp + tn) / total
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return accuracy, sensitivity, specificity, {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def call(self, x, *args, **kwargs):
        votes = [model.predict_cont(xi, *args, **kwargs) for model, xi in zip(self.models, x)]
        votes = np.stack(votes, axis=1)
        votes = np.mean(votes, axis=1)
        return np.array([1 if vote >= 0.5 else 0 for vote in votes])

