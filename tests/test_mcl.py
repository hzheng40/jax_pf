from jax_pf.src.mcl import *
from jax_pf.src.ray_marching import edt

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pathlib
from collections import namedtuple
import requests
import tarfile
from PIL import Image
import tempfile
import yaml

Trajectory = namedtuple("Trajectory", ["poses", "scans", "actions"])
MapInfo = namedtuple("MapInfo", ["omap", "resolution", "origin"])


@pytest.fixture
def traj():
    file = pathlib.Path("tests/test_data/test_traj.npz")
    archive = np.load(file)
    traj = Trajectory(
        poses=archive["poses"], scans=archive["scans"], actions=archive["actions"]
    )
    return traj


@pytest.fixture
def dt():
    map_url = "http://api.f1tenth.org/Spielberg.tar.xz"
    map_r = requests.get(url=map_url, allow_redirects=True)
    if map_r.status_code == 404:
        raise FileNotFoundError(f"URL wrong.")
    tempdir = tempfile.gettempdir() + "/"
    with open(tempdir + "Spielberg.tar.xz", "wb") as f:
        f.write(map_r.content)
    map_f = tarfile.open(tempdir + "Spielberg.tar.xz")
    map_f.extractall(tempdir)
    map_f.close()

    map_img = Image.open(tempdir + "Spielberg/Spielberg_map.png").transpose(
        Image.Transpose.FLIP_TOP_BOTTOM
    )
    omap = jnp.array(map_img).astype(jnp.float32)
    omap.at[omap <= 128].set(0.0)
    omap.at[omap > 128].set(255.0)

    with open(tempdir + "Spielberg/Spielberg_map.yaml") as f:
        map_metadata = yaml.safe_load(f)
    resolution = map_metadata["resolution"]
    origin = jnp.array(map_metadata["origin"])

    return MapInfo(omap=resolution * edt(omap), resolution=resolution, origin=origin)


def test_sensor_table():
    z_short = 0.01
    z_max = 0.07
    z_rand = 0.12
    z_hit = 0.75
    sigma_hit = 8.0
    max_range_px = int(30 / 0.05796)  # Spielberg
    table = compute_sensor_model(z_short, z_max, z_rand, z_hit, sigma_hit, max_range_px)

    import matplotlib.pyplot as plt

    plt.imshow(table)
    plt.show()


def test_motion_update():
    rng = jax.random.PRNGKey(0)
    # x, y, theta
    starting_pose = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.2], [3.0, 0.0, 0.0]])
    # steer, speed
    action = jnp.array([0.3, 5.0])
    new_poses, rng = motion_update(rng, starting_pose, action, 0.01, 0.0, 0.0, 0.0, 0.32)
    print(new_poses)

def test_sensor_update(traj, dt):
    index = 10
    num_particles = 10
    z_short = 0.01
    z_max = 0.07
    z_rand = 0.12
    z_hit = 0.75
    sigma_hit = 8.0
    theta_dis = 2000
    fov = 4.7
    num_beams = 1080
    angle_increment = fov / (num_beams - 1)
    theta_index_increment = theta_dis * angle_increment / (2 * jnp.pi)
    theta_arr = jnp.linspace(0.0, 2 * jnp.pi, num=theta_dis)
    sines = jnp.sin(theta_arr)
    cosines = jnp.cos(theta_arr)
    eps = 0.0001
    orig_x = dt.origin[0]
    orig_y = dt.origin[1]
    orig_c = jnp.cos(dt.origin[2])
    orig_s = jnp.sin(dt.origin[2])
    height = dt.omap.shape[0]
    width = dt.omap.shape[1]
    resolution = dt.resolution
    omap = dt.omap
    max_range = 30.0
    max_range_px = int(max_range / resolution)
    inv_squash_factor = 2.2

    omap = dt.omap
    scans = traj.scans
    poses = traj.poses
    true_pose = poses[index, :]
    true_scan = scans[index, :]

    rng = jax.random.PRNGKey(0)
    rng, noise_rng = jax.random.split(rng, 2)
    noises = jax.random.normal(noise_rng, (num_particles, 3))
    noises = noises.at[:, 2].multiply(0.2)
    noises = noises.at[0, :].multiply(0.0)
    particles = true_pose + noises

    sensor_table = compute_sensor_model(
        z_short, z_max, z_rand, z_hit, sigma_hit, max_range_px
    )

    weights = sensor_update(particles, true_scan, sensor_table, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, omap, max_range, inv_squash_factor)

    theta_index = theta_dis * (particles[:, 2] - fov / 2.0) / (2.0 * jnp.pi)
    # make sure it's wrapped properly
    theta_index = jnp.fmod(theta_index, theta_dis).astype(int)
    # inc = lambda x: x + theta_dis
    # noinc = lambda x: x
    # theta_index = jax.lax.cond((theta_index < 0), inc, noinc, theta_index)

    theta_indices = jnp.linspace(
        start=theta_index,
        stop=theta_index + theta_index_increment * num_beams,
        num=num_beams,
        endpoint=True,
        dtype=int,
    )

    thetas = theta_arr[theta_indices].T

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
        particles,
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
        omap,
        max_range,
    )

    print(np.array(weights))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(1, num_particles):
        ax.scatter(thetas[i, :], scans[i, :], alpha=float(weights[i]))
    ax.scatter(thetas[0, :],  true_scan)
    plt.show()


def test_localization(traj, dt):
    theta_dis = 2000
    fov = 4.7
    num_beams = 1080
    angle_increment = fov / (num_beams - 1)
    theta_index_increment = theta_dis * angle_increment / (2 * jnp.pi)
    theta_arr = jnp.linspace(0.0, 2 * jnp.pi, num=theta_dis)
    sines = jnp.sin(theta_arr)
    cosines = jnp.cos(theta_arr)
    eps = 0.0001
    orig_x = dt.origin[0]
    orig_y = dt.origin[1]
    orig_c = jnp.cos(dt.origin[2])
    orig_s = jnp.sin(dt.origin[2])
    orig_t = dt.origin[2]
    height = dt.omap.shape[0]
    width = dt.omap.shape[1]
    resolution = dt.resolution
    omap = dt.omap
    max_range = 30.0

    z_short = 0.01
    z_max = 0.07
    z_rand = 0.12
    z_hit = 0.75
    sigma_hit = 8.0
    max_range_px = int(max_range / resolution)  # Spielberg
    num_particles = 200

    motion_dispersion_x = 0.05
    motion_dispersion_y = 0.025
    motion_dispersion_t = 0.25

    inv_squash_factor = 2.2

    lwb = 0.32

    sensor_table = compute_sensor_model(
        z_short, z_max, z_rand, z_hit, sigma_hit, max_range_px
    )

    all_poses = traj.poses
    all_actions = traj.actions
    all_scans = traj.scans

    scan_downsample_step = int(all_scans.shape[1] / num_beams) + 1

    # init pf
    rng = jax.random.PRNGKey(0)
    # particles, weights, rng = mcl_init(
    #     rng, omap, num_particles, orig_x, orig_y, orig_c, orig_s, orig_t, resolution
    # )
    particles, weights, rng = mcl_init_with_pose(rng, all_poses[0, :], num_particles)

    all_pf_xs = []
    all_pf_ys = []
    all_pf_ts = []

    for i in range(all_poses.shape[0]):
        action = all_actions[i, :]
        true_pose = all_poses[i, :]
        scan = all_scans[i, :]
        # downsampled_scan = scan[::scan_downsample_step]

        particles, weights, rng = mcl_update(
            rng,
            particles,
            weights,
            action,
            scan,
            0.01,
            motion_dispersion_x,
            motion_dispersion_y,
            motion_dispersion_t,
            lwb,
            sensor_table,
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
            omap,
            max_range,
            inv_squash_factor,
        )
        all_pf_xs.append(particles[:, 0])
        all_pf_ys.append(particles[:, 1])
        all_pf_ts.append(particles[:, 2])
    
    
    all_pf_xs = np.array(all_pf_xs)
    all_pf_ys = np.array(all_pf_ys)
    all_pf_ts = np.array(all_pf_ts)
    
    import matplotlib.pyplot as plt
    
    ax1 = plt.subplot(311)
    plt.plot(np.arange(all_pf_xs.shape[0]), all_pf_xs, alpha=0.2)
    plt.plot(np.arange(all_pf_xs.shape[0]), all_poses[:, 0])
    # ax1.title("all particle xs vs true x")

    # share x only
    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(np.arange(all_pf_ys.shape[0]), all_pf_ys, alpha=0.2)
    plt.plot(np.arange(all_pf_ys.shape[0]), all_poses[:, 1])
    # ax2.title("all particle ys vs true y")

    # share x and y
    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(np.arange(all_pf_ts.shape[0]), all_pf_ts, alpha=0.2)
    plt.plot(np.arange(all_pf_ts.shape[0]), all_poses[:, 2])
    # ax3.title("all particle thetas vs true theta")
    
    plt.show()

