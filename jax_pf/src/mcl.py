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


@partial(jax.jit, static_argnums=[5])
@chex.assert_max_traces(n=2)
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
        maximum scan range in pixels

    Returns
    -------
    ArrayLike
        _description_
    """
    sensor_model_table = jnp.zeros((max_range_px + 1, max_range_px + 1))
    # d is the computed range from rm
    drange = jnp.arange(max_range_px + 1)
    # r is the observed range from lidar
    rrange = jnp.arange(max_range_px + 1)
    dm, rm = jnp.meshgrid(drange, rrange)
    dr = jnp.stack((dm.flatten(), rm.flatten()))
    d = dr[0, :]
    r = dr[1, :]
    z = r - d

    prob = (
        z_hit
        * jnp.exp(-(z**2) / (2.0 * sigma_hit**2))
        / (sigma_hit * jnp.sqrt(2.0 * jnp.pi))
    )
    prob = jax.lax.select((r < d), prob + 2 * z_short * (d - r) / d, prob)
    prob = jax.lax.select((r == max_range_px), prob + z_max, prob)
    prob = jax.lax.select((r < max_range_px), prob + z_rand / max_range_px, prob)
    sensor_model_table = sensor_model_table.at[r, d].set(prob)

    # normalize each row
    row_sum = jnp.sum(sensor_model_table, axis=1)
    sensor_model_table = sensor_model_table.at[drange, :].divide(row_sum)

    return sensor_model_table


def motion_update(particle_state: Array, action: Array) -> ArrayLike:
    # TODO: motion update, vmapped over all particles
    # TODO: returns updated state
    pass


def sensor_update(particle_state: Array, observation: Array) -> ArrayLike:
    # TODO: sensor update, vmapped over all particles
    # TODO: returns weighting for each particle
    pass


def mcl_init(
    permissible: Array,
    max_particles: int,
    orig_x: float,
    orig_y: float,
    orig_t: float,
    resolution: float,
) -> ArrayLike:
    # TODO: initialize global particle states
    pass


def mcl_update(particles: Array, action: Array, observation: Array) -> ArrayLike:
    # stateless MCL update
    # TODO: 1. motion update, all particles
    # TODO: 2. sensor update, all particles
    # TODO: 3. normalize all particle weights
    # TODO: 4. resampling with new weights
    # TODO: 5. draw new proposal distribution from particle weights
    #          and update proposal distribution
    pass
