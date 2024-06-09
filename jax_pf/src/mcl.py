from functools import partial
import os
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from jax import Array
from jax.typing import ArrayLike
import numpy as np

from .ray_marching import get_scan

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def compute_sensor_model(
    z_short: float,
    z_max: float,
    z_rand: float,
    z_hit: float,
    sigma_hit: float,
    max_range_px: int,
) -> ArrayLike:
    """Generate and store a table which represents the sensor model.
    For each discrete computed range value, this provides the probability of measuring any (discrete) range.

    Parameters
    ----------
    z_short : float
        _description_
    z_max : float
        _description_
    z_rand : float
        _description_
    z_hit : float
        _description_
    sigma_hit : float
        _description_
    max_range_px : int
        _description_

    Returns
    -------
    ArrayLike
        _description_
    """
    sensor_model_table = jnp.zeros((max_range_px + 1, max_range_px + 1))
    # d is the computed range from rm
    for d in range(max_range_px + 1):
        norm = 0.0
        # r is the observed range from lidar
        for r in range(max_range_px + 1):
            prob = 0.0
            z = r - d
            # reflects from the intented object
            prob += (
                z_hit
                * jnp.exp(-(z**2) / (2.0 * sigma_hit**2))
                / (sigma_hit * jnp.sqrt(2.0 * np.pi))
            )
            # observed range is less than predicted range (short reading)
            prob = jax.lax.select((r < d), prob + 2 * z_short * (d - r) / d, prob)
            # errorneous max range measurement
            prob = jax.lax.select((r == max_range_px), prob + z_max, prob)
            # random measurement
            prob = jax.lax.select(
                (r < max_range_px), prob + z_rand / max_range_px, prob
            )

            norm += prob
            sensor_model_table.at[r, d].set(prob)

        # normalize
        sensor_model_table.at[:, d].divide(norm)

    return sensor_model_table


def motion_update(particle_state: Array, action: Array) -> ArrayLike:
    pass


def sensor_update(particle_state: Array, observation: Array) -> ArrayLike:
    pass


def mcl_update(particles: Array, action: Array, observation: Array):

    pass
