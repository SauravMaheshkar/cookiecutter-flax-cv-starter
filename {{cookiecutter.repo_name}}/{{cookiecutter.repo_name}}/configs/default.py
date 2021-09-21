"""Default Hyperparameter configuration"""
import wandb


def get_config():

    config = wandb.config

    config.batch_size = 64
    config.num_epochs = 10
    config.lr_rate = 0.1
    config.momentum = 0.9

    return config
