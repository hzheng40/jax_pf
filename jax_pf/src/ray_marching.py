from functools import partial
import os
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from jax import Array
from jax.typing import ArrayLike

import numpy as np
from scipy.ndimage import distance_transform_edt as edt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@jax.jit
@chex.assert_max_traces(n=2)
def xy_2_rc(
    x: float,
    y: float,
    orig_x: float,
    orig_y: float,
    orig_c: float,
    orig_s: float,
    height: float,
    width: float,
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
    height : float
        occupancy map height
    width : float
        occupancy map width
    resolution : float
        occupancy map resolution in m/pixel

    Returns
    -------
    Tuple[int]
        row, column cooridnate
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if (
        x_rot < 0
        or x_rot >= width * resolution
        or y_rot < 0
        or y_rot >= height * resolution
    ):
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)

    return jnp.array([r, c])


@jax.jit
@chex.assert_max_traces(n=2)
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


@partial(jax.jit, static_argnums=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
@chex.assert_max_traces(n=2)
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

    init_dist = (dist_to_nearest, total_dist)

    def trace_step(dists):
        x += dists[0] * c
        y += dists[0] * s

        # update dist_to_nearest for current point on ray
        # also keeps track of total ray length
        dist_to_nearest = distance_transform(
            x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt
        )
        total_dist += dist_to_nearest
        return total_dist

    def trace_cond(dists):
        return dists[0] > eps and dists[1] <= max_range

    # ray tracing iterations
    total_dist = jax.lax.while_loop(trace_cond, trace_step, init_dist)

    jnp.clip(total_dist, max=max_range)

    return total_dist


@partial(
    jax.jit, static_argnums=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
)
@chex.assert_max_traces(n=2)
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
    # make theta discrete by mapping the range [-pi, pi] onto [0, theta_dis]
    theta_index = theta_dis * (pose[2] - fov / 2.0) / (2.0 * np.pi)

    # make sure it's wrapped properly
    theta_index = jnp.fmod(theta_index, theta_dis)
    if theta_index < 0:
        theta_index += theta_dis

    theta_indices = jnp.linspace(
        start=theta_index,
        stop=theta_index + theta_index_increment * num_beams,
        num=num_beams,
        endpoint=True,
        dtype=int,
    )[:, None]

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
