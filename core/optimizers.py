#    Author: Ankit Kariryaa, University of Bremen

from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Nadam


# Optimizers; https://keras.io/optimizers/
adaDelta = Adadelta(learning_rate=1.0, rho=0.95, epsilon=None, decay=0.0)
adam = Adam(lr= 5.0e-05, decay= 0.0, beta_1= 0.9, beta_2=0.999, epsilon= 1.0e-8)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)


def get_optimizer(optimizer_fn):
    """Wrapper function to allow only storing optimizer function name in config"""
    if optimizer_fn == "adaDelta":
        return adaDelta
    elif optimizer_fn == "adam":
        return adam
    elif optimizer_fn == "nadam":
        return nadam
    elif optimizer_fn == "adagrad":
        return adagrad
    else:
        # Used when passing string names of built-in tensorflow optimizers
        return optimizer_fn
