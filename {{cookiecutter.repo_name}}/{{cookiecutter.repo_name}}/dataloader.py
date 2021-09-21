# From https://github.com/google/flax/blob/main/examples/mnist/train.py
import jax.numpy as jnp
import tensorflow_datasets as tfds

__all__ = ["get_datasets"]


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
    return train_ds, test_ds
