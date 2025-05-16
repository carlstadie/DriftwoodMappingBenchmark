from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Nadam#, AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow as tf

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class WarmUpCosineDecay(LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, final_lr_scale=0.1):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.final_lr = base_lr * final_lr_scale
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=final_lr_scale
        )

    def __call__(self, step):
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.cosine_decay(step - self.warmup_steps)
        )



# Optimizers; https://keras.io/optimizers/
adaDelta = Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-7, decay=0.0)
adam = Adam(lr= 3e-4, decay= 0.0, beta_1= 0.9, beta_2=0.999, epsilon= 1.0e-8, clipnorm=1.0)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
#adamw = AdamW(learning_rate=1e-4, weight_decay=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0)


def get_optimizer(optimizer_fn, num_epochs=None, steps_per_epoch=None):
    """Wrapper function to allow dynamic optimizer setup with optional learning rate schedule"""
    
    if optimizer_fn == "adam":
        if num_epochs and steps_per_epoch:
            total_steps = num_epochs * steps_per_epoch
            lr_schedule = WarmUpCosineDecay(base_lr=3e-4, total_steps=total_steps, warmup_steps=1000)

            return Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0)
        else:
            return Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0)
    elif optimizer_fn == "adam1":
        return adam

    elif optimizer_fn == "adaDelta":
        return Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-7)
    elif optimizer_fn == "nadam":
        return Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
    elif optimizer_fn == "adagrad":
        return Adagrad(learning_rate=0.01)
    else:
        return optimizer_fn