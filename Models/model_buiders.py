from tensorflow import keras

from Models.ESN.ESN import ESN
from Models.IRESN.IRESN import IRESN
from Models.TrainableESN.TrainableESN import TrainableESN
from Models.TrainableIRESN.TrainableIRESN import TrainableIRESN


def build_ESN(hp):
    model = ESN(units=hp["units"],
                input_shape=hp["input_shape"],
                leaky=hp["leaky"],
                input_scaling=hp["input_scaling"],
                bias_scaling=hp["bias_scaling"],
                spectral_radius=hp["spectral_radius"],
                reg=hp["reg"])
    return model


def build_IRESN(hp):
    model = IRESN(units=hp["units"],
                  leaky=hp["leaky"],
                  sub_reservoirs=hp["sub_reservoirs"],
                  input_scaling=hp["input_scaling"],
                  bias_scaling=hp["bias_scaling"],
                  spectral_radius=hp["spectral_radius"],
                  lasso=False)
    return model


def build_IRESN_Bayes(hp):
    hp["input_scaling"] = [hp[f"input_scaling{i+1}"] for i in range(hp["sub_reservoirs"])]
    hp["bias_scaling"] = [hp[f"bias_scaling{i+1}"] for i in range(hp["sub_reservoirs"])]
    if "units_sub1" in hp:
        sub_sum = sum([hp[f"units_sub{i + 1}"] for i in range(hp["sub_reservoirs"])])
        hp["units"] = [int(hp[f"units_sub{i + 1}"] / sub_sum * hp["units"]) for i in range(hp["sub_reservoirs"])]
    return build_IRESN(hp)


def build_TrainableESN(hp):
    model = TrainableESN(units=hp["units"],
                         leaky=hp["leaky"],
                         input_scaling=hp["input_scaling"],
                         bias_scaling=hp["bias_scaling"],
                         spectral_radius=hp["spectral_radius"],
                         connectivity=hp["connectivity"])

    model.compile(optimizer=keras.optimizers.Adam(hp["learning_rate"]),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model


def build_TrainableIRESN(hp):
    model = TrainableIRESN(units=hp["units"],
                           sub_reservoirs=hp["sub_reservoirs"],
                           leaky=hp["leaky"],
                           input_scaling=hp["input_scaling"],
                           bias_scaling=hp["bias_scaling"],
                           inter_scaling=hp["inter_scaling"],
                           spectral_radius=hp["spectral_radius"])

    model.compile(optimizer=keras.optimizers.Adam(hp["learning_rate"]),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model
