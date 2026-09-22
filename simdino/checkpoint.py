"""Checkpoint I/O stays outside the compiled training algorithm."""
import pickle
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp

from config import Config
from train import State


def save(path: Path, state: State, config: Config):
    temporary = path.with_suffix(".tmp")
    payload = {"config": asdict(config), "state": jax.device_get(state)}
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream)
    temporary.replace(path)


def load(path: Path) -> tuple[Config, State]:
    # Pickle checkpoints are for trusted local files, not arbitrary downloads.
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    state = State(*jax.tree.map(jnp.asarray, payload["state"]))
    return Config(**payload["config"]), state
