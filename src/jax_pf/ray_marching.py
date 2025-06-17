from functools import partial
import os

import jax
import jax.numpy as jnp
import chex
from jax import Array
from jax.typing import ArrayLike

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@jax.jit
@chex.assert_max_traces(n=1)
def xy_2_rc(
    x: float,
    y: float,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    height: int,
    width: int,
    resolution: float,
) -> ArrayLike:
    """Transform x, y coordinates to row, column coordinates in occupancy map

    Parameters
    ----------
    x : float
        x position
    y : float
        y position
    orig_x : float
        x position of map origin
    orig_y : float
        y position of map origin
    orig_c : float
        cosine of map origin rotation
    orig_s : float
        sine of map origin rotation
    height : int
        occupancy map height
    width : int
        occupancy map width
    resolution : float
        occupancy map resolution in m/pixel

    Returns
    -------
    ArrayLike
        row, column cooridnate
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell:
    r = jnp.clip(y_rot / resolution, a_min=0, a_max=height)
    c = jnp.clip(x_rot / resolution, a_min=0, a_max=width)

    rc = jnp.array([r, c], dtype=int)

    return rc


@jax.jit
@chex.assert_max_traces(n=1)
def rc_2_xy(
    rc: Array,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    orig_theta: float,
    resolution: float,
) -> ArrayLike:
    """Transform row column coordinates to x, y coordinates in world

    Parameters
    ----------
    rc : Array
        row column coordinates
    orig_x : float
        x position of map origin
    orig_y : float
        y position of map origin
    orig_c : float
        cosine of map origin rotation
    orig_s : float
        sine of map origin rotation
    orig_theta : float
        map origin rotation
    resolution : float
        occupancy map resolution in m/pixel

    Returns
    -------
    ArrayLike
        x, y, theta cooridnate
    """
    # rotation
    x_rot = orig_c * rc[:, 0] - orig_s * rc[:, 1]
    y_rot = orig_s * rc[:, 0] + orig_c * rc[:, 1]

    x = x_rot * resolution + orig_x
    y = y_rot * resolution + orig_y
    theta = rc[:, 2] + orig_theta

    xyt = jnp.vstack((x, y, theta)).T

    return xyt


@jax.jit
@chex.assert_max_traces(n=1)
def distance_transform(
    x: float,
    y: float,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    height: float,
    width: float,
    resolution: float,
    dt: ArrayLike,
) -> float:
    """Look up corresponding distance in the distance transform matrix

    Parameters
    ----------
    x : float
        x position
    y : float
        y position
    orig_x : float
        x position of map origin
    orig_y : float
        y position of map origin
    orig_c : float
        cosine of map origin rotation
    orig_s : float
        sine of map origin rotation
    height : float
        occupancy map height
    width : float
        occupancy map width
    resolution : float
        occupancy map resolution in m/pixel
    dt : jax.Array
        euclidean distance transform matrix

    Returns
    -------
    float
        corresponding shortest distance to obstacle in meters
    """
    r, c = xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution)
    distance = dt[r, c]
    return distance


@partial(jax.jit, static_argnums=[5, 14])
@chex.assert_max_traces(n=1)
def trace_ray(
    x: float,
    y: float,
    theta_index: int,
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
) -> float:
    """Find the length of a specific ray at a specific scan angle theta

    Parameters
    ----------
    x : float
        x position of ray start
    y : float
        y position of ray start
    theta_index : int
        index of ray start direction in preset angle array
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
    float
        length of traced ray
    """
    # int casting, and index precal trigs
    s = sines[theta_index]
    c = cosines[theta_index]

    # distance to nearest initialization
    dist_to_nearest = distance_transform(
        x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt
    )
    total_dist = dist_to_nearest

    init_dist = jnp.array([dist_to_nearest, total_dist, x, y])

    def trace_step(dists):
        x = (dists[2] + dists[0] * c)[0]
        y = (dists[3] + dists[0] * s)[0]

        # update dist_to_nearest for current point on ray
        # also keeps track of total ray length
        dist_to_nearest = distance_transform(
            x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt
        )
        total_dist = dists[1] + dist_to_nearest
        return jnp.array([dist_to_nearest, total_dist, x, y])

    def trace_cond(dists):
        return (dists[0] > eps) & (dists[1] <= max_range)

    # ray tracing iterations
    final_dist = jax.lax.while_loop(trace_cond, trace_step, init_dist)
    total_dist = jnp.clip(final_dist[1], a_max=max_range)

    return total_dist


@partial(jax.jit, static_argnums=[3])
@chex.assert_max_traces(n=1)
def get_scan(
    pose: ArrayLike,
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
) -> Array:
    """Perform the scan for all discretized angle of all beam of the laser

    Parameters
    ----------
    pose : ArrayLike
        current pose of the scan frame in the map, (x, y, theta)
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
    Array
        resulting laser scan at the pose, with length num_beams
    """
    theta_index_start = theta_dis * (pose[2] - fov / 2.0) / (2.0 * jnp.pi)
    theta_indices = jnp.linspace(
        start=theta_index_start,
        stop=theta_index_start + theta_index_increment * num_beams,
        num=num_beams,
        endpoint=True,
        dtype=int,
    )[:, None]
    # make sure it's wrapped properly
    theta_indices = jax.lax.select(
        theta_indices < 0, theta_indices + theta_dis, theta_indices
    )
    theta_indices = jnp.fmod(theta_indices, theta_dis)

    # vmap to vectorize each ray march
    # vectorized over multiple theta_index inputs
    trace_ray_vmap = jax.vmap(
        trace_ray,
        (
            None,
            None,
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
        ),
    )
    scan = trace_ray_vmap(
        pose[0],
        pose[1],
        theta_indices,
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

    return scan
