from functools import partial
import os
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import chex
from jax import Array
from jax.typing import ArrayLike
import numpy as np

from .ray_marching import get_scan, rc_2_xy

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


@jax.jit
@chex.assert_max_traces(n=2)
def motion_update(
    rng: PRNGKey,
    particle_state: Array,
    action: Array,
    dispersion_x: float,
    dispersion_y: float,
    dispersion_t: float,
    lwb: float,
) -> Tuple[ArrayLike, PRNGKey]:
    """Motion update of all particles, uses kinematics model

    Parameters
    ----------
    rng : PRNGKey
        rng key
    particle_state : Array
        current particle states
    action : Array
        action taken at previous step
    dispersion_x : float
        x coordinate motion dispersion
    dispersion_y : float
        y coordinate motion dispersion
    dispersion_t : float
        yaw coordinate motion dispersion
    lwb : float
        vehicle wheel base

    Returns
    -------
    Tuple[ArrayLike, PRNGKey]
        1. Motion updated particle states
        2. The rng key after split
    """

    # kinematic model, action is (speed, steer)
    cosines = jnp.cos(particle_state[:, 2])
    sines = jnp.sin(particle_state[:, 2])
    particle_state = particle_state.at[:, 0].add(cosines * action[0])
    particle_state = particle_state.at[:, 1].add(sines * action[0])
    particle_state = particle_state.at[:, 2].add((action[1] / lwb) * jnp.tan(action[1]))
    # add noise
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, particle_state.shape)
    noise = noise.at[:, 0].multiply(dispersion_x)
    noise = noise.at[:, 1].multiply(dispersion_y)
    noise = noise.at[:, 2].multiply(dispersion_t)

    particle_state = particle_state + noise

    return particle_state, rng


@jax.jit
@chex.assert_max_traces(n=2)
def sensor_update(
    particle_state: Array,
    observation: Array,
    sensor_model_table: Array,
    theta_dis: int,
    fov: float,
    num_beams: int,
    theta_index_increment: float,
    sines: ArrayLike,
    cosines: ArrayLike,
    eps: float,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    height: float,
    width: float,
    resolution: float,
    dt: ArrayLike,
    max_range: float,
    inv_squash_factor: float,
) -> ArrayLike:
    """Sensor update of all particles, returns weighting for each particle based on actual observation

    Parameters
    ----------
    particle_state : Array
        current particle states
    observation : Array
        current actual scan observation
    sensor_model_table : Array
        precomputed sensor model table
    theta_dis : int
        number of steps to discretize the angles between 0 and 2pi for look up
    fov : float
        field of view of the laser scan
    num_beams : int
        number of beams in the scan
    theta_index_increment : float
        increment between angle indices after discretization
    sines : ArrayLike
        sines of preset angle array
    cosines : ArrayLike
        cosines of preset angle array
    eps : float
        ray trace ending tolerance
    orig_x : float
        x of map origin
    orig_y : float
        y of map origin
    orig_c : float
        cosine of rotation of map origin
    orig_s : float
        sine of rotation of map origin
    height : float
        height of map
    width : float
        width of map
    resolution : float
        resolution of map
    dt : ArrayLike
        euclidean distance transform of map
    max_range : float
        maximum ray lenth
    inv_squash_factor : float
        particle weight squashing factor

    Returns
    -------
    ArrayLike
        updated weight for each particle
    """

    # TODO: 1. calculate scans of all particles
    get_scan_vmapped = jax.vmap(
        get_scan,
        in_axes=[
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    )
    scans = get_scan_vmapped(
        particle_state,
        theta_dis,
        fov,
        num_beams,
        theta_index_increment,
        sines,
        cosines,
        eps,
        orig_x,
        orig_y,
        orig_c,
        orig_s,
        height,
        width,
        resolution,
        dt,
        max_range,
    )

    print(scans.shape)  # TODO: should be (num_particles, num_beams)

    # TODO: 2. resolve sensor model, discretize and index into precomputed table
    max_range_px = int(max_range / resolution)
    observation = observation / resolution
    scans = scans / resolution
    observation.at[observation > max_range_px].set(max_range_px)
    scans.at[scans > max_range_px].set(max_range_px)

    intobservation = jnp.rint(observation).astype(jnp.uint16)
    intscans = jnp.rint(scans).astype(jnp.uint16)

    @partial(jax.vmap, in_axes=[None, 0, None])
    def get_weight(table, pscan, obs):
        weight = jnp.power(jnp.prod(table[obs, pscan]), inv_squash_factor)

    weights = get_weight(sensor_model_table, intscans, intobservation)

    print(weights.shape)  # TODO: should be (num_particles, 1)

    return weights


@jax.jit
@chex.assert_max_traces(n=2)
def mcl_init(
    rng: PRNGKey,
    omap: Array,
    num_particles: int,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    orig_theta: float,
    resolution: float,
) -> Tuple[ArrayLike, ArrayLike, PRNGKey]:
    """Initialize global particle states in permissible region

    Parameters
    ----------
    rng : PRNGKey
        rng key
    omap : Array
        occupancy map
    num_particles : int
        number of particles
    orig_x : float
        x coordinate of map origin
    orig_y : float
        y coordinate of map origin
    orig_c : float
        cosine of map origin rotation
    orig_s : float
        sine of map origin rotation
    orig_theta : float
        theta coordinate of map origin
    resolution : float
        map resolution in meters/pixel

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, PRNGKey]
        1. particle states
        2. weights of particles
        3. rng key after split
    """

    rng, ind_rng, heading_rng = jax.random.split(rng, 3)

    permissible_ind_r, permissible_ind_c = jnp.where(omap == 0)
    chosen_ind = jax.random.randint(
        ind_rng, shape=(num_particles,), minval=0, maxval=len(permissible_ind_r)
    )
    random_headings = jax.random.uniform(
        heading_rng, shape=(num_particles,), minval=0.0, maxval=2 * jnp.pi
    )

    # particles in pixel coordinates
    particles = jnp.zeros((num_particles, 3))
    particles = particles.at[:, 0].set(permissible_ind_c[chosen_ind])
    particles = particles.at[:, 1].set(permissible_ind_r[chosen_ind])
    particles = particles.at[:, 2].set(random_headings)

    # transform particles to world coordinates
    particles = rc_2_xy(
        particles, orig_x, orig_y, orig_c, orig_s, orig_theta, resolution
    )

    # initialize weight for each particle uniformly
    weights = jnp.ones(num_particles) / num_particles

    return particles, weights, rng


@jax.jit
@chex.assert_max_traces(n=2)
def mcl_init_with_pose(
    rng: PRNGKey,
    pose: Array,
    num_particles: int,
) -> Tuple[ArrayLike, ArrayLike, PRNGKey]:
    """Initialize global particle states around given pose

    Parameters
    ----------
    rng : PRNGKey
        rng key
    pose : Array
        given initialization pose
    num_particles : int
        number of particles

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, PRNGKey]
        1. particle states
        2. weights of particles
        3. rng key after split
    """
    rng, sample_rng = jax.random.split(rng, 2)

    # noise around initialization
    noises = jax.random.normal(sample_rng, (num_particles, 3)) * 0.5

    # particles state
    particles = noises + pose

    # TODO should be (num_particles, 3)
    print(particles.shape)

    # initialize weight for each particle uniformly
    weights = jnp.ones(num_particles) / num_particles

    return particles, weights, rng


@jax.jit
@chex.assert_max_traces(n=2)
def mcl_update(
    rng: PRNGKey,
    particles: Array,
    weights: Array,
    action: Array,
    observation: Array,
    dispersion_x: float,
    dispersion_y: float,
    dispersion_t: float,
    lwb: float,
    sensor_model_table: Array,
    theta_dis: int,
    fov: float,
    num_beams: int,
    theta_index_increment: float,
    sines: ArrayLike,
    cosines: ArrayLike,
    eps: float,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    height: int,
    width: int,
    resolution: float,
    dt: ArrayLike,
    max_range: float,
    inv_squash_factor: float,
) -> Tuple[ArrayLike, ArrayLike, PRNGKey]:
    """Stateless mcl update step

    Parameters
    ----------
    rng : PRNGKey
        rng key
    particles : Array
        current particle state
    weights : Array
        current particle weights
    action : Array
        action taken at previous step
    observation : Array
        current actual scan observation
    dispersion_x : float
        x coordinate motion dispersion
    dispersion_y : float
        y coordinate motion dispersion
    dispersion_t : float
        yaw coordinate motion dispersion
    lwb : float
        vehicle wheel base
    sensor_model_table : Array
        precomputed sensor model table
    theta_dis : int
        number of steps to discretize the angles between 0 and 2pi for look up
    fov : float
        field of view of the laser scan
    num_beams : int
        number of beams in the scan
    theta_index_increment : float
        increment between angle indices after discretization
    sines : ArrayLike
        sines of preset angle array
    cosines : ArrayLike
        cosines of preset angle array
    eps : float
        ray trace ending tolerance
    orig_x : float
        x of map origin
    orig_y : float
        y of map origin
    orig_c : float
        cosine of rotation of map origin
    orig_s : float
        sine of rotation of map origin
    height : float
        height of map
    width : float
        width of map
    resolution : float
        resolution of map
    dt : ArrayLike
        euclidean distance transform of map
    max_range : float
        maximum ray lenth
    inv_squash_factor : float
        particle weight squashing factor

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, PRNGKey]
        New particle state, weights, and rng key after split
    """
    # stateless MCL update
    rng, redraw_rng = jax.random.split(rng, 2)

    # 0. redraw particles based on previous weight
    proposal_ind = jax.random.choice(redraw_rng, a=particles.shape[0], p=weights)
    proposal_particles = particles[proposal_ind, :]

    # 1. motion update, all particles
    proposal_particles, rng = motion_update(
        rng, proposal_particles, action, dispersion_x, dispersion_y, dispersion_t, lwb
    )
    # 2. sensor update, all particles
    new_weights = sensor_update(
        proposal_particles,
        observation,
        sensor_model_table,
        theta_dis,
        fov,
        num_beams,
        theta_index_increment,
        sines,
        cosines,
        eps,
        orig_x,
        orig_y,
        orig_c,
        orig_s,
        height,
        width,
        resolution,
        dt,
        max_range,
        inv_squash_factor,
    )
    # 3. normalize all particle weights
    new_weights = new_weights / jnp.sum(new_weights)

    # 4. return new particle state and new weights
    return proposal_particles, new_weights, rng