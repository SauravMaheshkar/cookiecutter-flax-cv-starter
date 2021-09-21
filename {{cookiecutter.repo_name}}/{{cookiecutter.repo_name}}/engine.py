# From https://github.com/google/flax/blob/main/examples/mnist/train.py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.metrics import tensorboard
from flax.training import train_state

from .dataloader import get_datasets
from .nn import Model
from .nn_utils import apply_model, update_model

__all__ = ["train_and_evaluate"]


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        wandb.log({"Training Loss": loss})
        epoch_accuracy.append(accuracy)
        wandb.log({"Training Accuracy": loss})
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    model = Model()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(config["learning_rate"], config["momentum"])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_evaluate(config: dict, workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config["num_epochs"] + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config["batch_size"], input_rng
        )
        _, test_loss, test_accuracy = apply_model(
            state, test_ds["image"], test_ds["label"]
        )

        print("epoch: %3d" % (epoch))
        print("train_loss: %.4f" % (train_loss))
        print("train_accuracy: %.2f" % (train_accuracy * 100))
        print("test_loss: %.4f" % (test_loss))
        print("test_accuracy: %.2f" % (test_accuracy * 100))

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
        summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    summary_writer.flush()
    return state
