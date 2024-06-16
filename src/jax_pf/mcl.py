from functools import partial
import os
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import chex
from jax import Array
from jax.typing import ArrayLike

from .ray_marching import get_scan, rc_2_xy

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@partial(jax.jit, static_argnums=[6])
@chex.assert_max_traces(n=1)
def compute_sensor_model(
    z_short: float,
    z_max: float,
    z_rand: float,
    z_hit: float,
    sigma_hit: float,
    lambda_short: float,
    max_range_px: int,
) -> ArrayLike:
    """Generate and store a table which represents the sensor model.
    For each discrete computed range value, this provides the probability of measuring any (discrete) range.
    Probablistic Robotics Chapter 6.3.1

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
    lambda_short : float
        short exponentioal dist. param
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

    # normal
    prob = (
        z_hit
        * jnp.exp(-(z**2) / (2.0 * sigma_hit**2))
        / (sigma_hit * jnp.sqrt(2.0 * jnp.pi))
    )
    # short exponential
    prob = jax.lax.select(
        (r < d),
        prob
        + z_short
        * (
            (lambda_short * jnp.exp(-lambda_short * r))
            / (1 - jnp.exp(-lambda_short * d))
        ),
        prob,
    )
    # sensor failures (measuring max instead of actual value)
    prob = jax.lax.select((r == max_range_px), prob + z_max, prob)
    # random measurement noise
    prob = jax.lax.select((r < max_range_px), prob + z_rand / max_range_px, prob)
    sensor_model_table = sensor_model_table.at[r, d].set(prob)

    # normalize each col
    col_sum = jnp.sum(sensor_model_table, axis=0)
    sensor_model_table = sensor_model_table.at[:, drange].divide(col_sum)

    return sensor_model_table


@jax.jit
@chex.assert_max_traces(n=1)
def motion_update(
    rng: PRNGKey,
    particle_state: Array,
    action: Array,
    dispersion_x: float,
    dispersion_y: float,
    dispersion_t: float,
) -> Tuple[ArrayLike, PRNGKey]:
    """Motion update of all particles, uses kinematics model

    Parameters
    ----------
    rng : PRNGKey
        rng key
    particle_state : Array
        current particle states
    action : Array
        change in odometry (accumulated with kinematic model)
    dispersion_x : float
        x coordinate motion dispersion
    dispersion_y : float
        y coordinate motion dispersion
    dispersion_t : float
        yaw coordinate motion dispersion

    Returns
    -------
    Tuple[ArrayLike, PRNGKey]
        1. Motion updated particle states
        2. The rng key after split
    """

    # cos/sin from odom
    cosines = jnp.cos(particle_state[:, 2])
    sines = jnp.sin(particle_state[:, 2])

    # add noise
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, particle_state.shape)

    # motion update
    particle_state_x = (
        particle_state[:, 0]
        + cosines * action[0]
        - sines * action[1]
        + noise[:, 0] * dispersion_x
    )
    particle_state_y = (
        particle_state[:, 1]
        + sines * action[0]
        + cosines * action[1]
        + noise[:, 1] * dispersion_y
    )
    particle_state_t = particle_state[:, 2] + action[2] + noise[:, 2] * dispersion_t

    particle_state = jnp.column_stack(
        (particle_state_x, particle_state_y, particle_state_t)
    )

    return particle_state, rng


@partial(jax.jit, static_argnums=[5])
@chex.assert_max_traces(n=1)
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

    Returns
    -------
    ArrayLike
        updated weight for each particle
    """

    # 1. calculate scans of all particles
    get_scan_vmapped = jax.jit(
        jax.vmap(
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
        ),
        static_argnums=[3],
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

    # 2. resolve sensor model, discretize and index into precomputed table
    max_range_px = (max_range / resolution).astype(int)
    observation = observation / resolution
    scans = scans / resolution
    # clip scans by max range
    observation = jnp.clip(observation, a_max=max_range_px)
    scans = jnp.clip(scans, a_max=max_range_px)

    intobservation = jnp.rint(observation).astype(int)
    intscans = jnp.rint(scans).astype(int)

    @partial(jax.vmap, in_axes=[None, 0, None])
    def get_weight(table, pscan, obs):
        weight = jnp.sum(jnp.log(table[obs, pscan]))
        return weight

    weights = get_weight(sensor_model_table, intscans, intobservation)
    weights = weights - weights.min()
    weights /= weights.max() + 0.0000001
    return weights


@partial(jax.jit, static_argnums=[2])
@chex.assert_max_traces(n=1)
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

    # permissible feature cannot be supported by jax
    # have to initialize over entire map
    permissible_ind_r = jnp.arange(omap.shape[0], dtype=int)
    permissible_ind_c = jnp.arange(omap.shape[1], dtype=int)
    chosen_ind = jax.random.randint(
        ind_rng, shape=(num_particles, 1), minval=0, maxval=len(permissible_ind_r)
    ).flatten()
    random_headings = jax.random.uniform(
        heading_rng, shape=(num_particles, 1), minval=0.0, maxval=2 * jnp.pi
    ).flatten()

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


@partial(jax.jit, static_argnums=[2])
@chex.assert_max_traces(n=1)
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

    # initialize weight for each particle uniformly
    weights = jnp.ones(num_particles) / num_particles

    return particles, weights, rng


@partial(jax.jit, static_argnums=[11])
@chex.assert_max_traces(n=1)
def mcl_update(
    rng: PRNGKey,
    particles: Array,
    weights: Array,
    action: Array,
    observation: Array,
    dispersion_x: float,
    dispersion_y: float,
    dispersion_t: float,
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
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, PRNGKey]:
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

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, ArrayLike, PRNGKey]
        New particle state, weights, current estimate, and rng key after split
    """
    # stateless MCL update
    rng, redraw_rng = jax.random.split(rng, 2)

    # # renormalize weights
    # weights = weights / jnp.sum(weights)

    # 0. redraw particles based on previous weight
    proposal_ind = jax.random.choice(
        redraw_rng, a=particles.shape[0], shape=(particles.shape[0], 1), p=weights
    ).flatten()
    # proposal_ind = jax.random.choice(
    #     redraw_rng, a=particles.shape[0], shape=(particles.shape[0], 1)).flatten()
    proposal_particles = particles[proposal_ind, :]

    # 1. motion update, all particles
    proposal_particles, rng = motion_update(
        rng,
        proposal_particles,
        action,
        dispersion_x,
        dispersion_y,
        dispersion_t,
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
    )
    # 3. normalize all particle weights
    new_weights = new_weights / jnp.sum(new_weights)

    # 4. return new particle state and new weights and current estimate
    current_estimate = jnp.dot(new_weights, proposal_particles)
    return proposal_particles, new_weights, current_estimate, rng
