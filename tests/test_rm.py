from jax_pf.src.ray_marching import *
import pytest
import requests
import pathlib
import tarfile
import tempfile
import yaml
from collections import namedtuple
from PIL import Image

MapInfo = namedtuple("MapInfo", ["omap", "resolution", "origin"])


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

@pytest.fixture
def ref_scan():
    file = pathlib.Path("tests/test_data/test_scan.txt")
    scan_to_match = jnp.array(np.loadtxt(file))
    return scan_to_match


def test_scan(dt, ref_scan):
    pose = jnp.array([-0.0440806, -0.8491629, 3.4034119])
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
    scan = get_scan(
        pose,
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

    theta_index = theta_dis * (pose[2] - fov / 2.0) / (2.0 * jnp.pi)

    # make sure it's wrapped properly
    theta_index = jnp.fmod(theta_index, theta_dis)
    inc = lambda x: x + theta_dis
    noinc = lambda x: x
    theta_index = jax.lax.cond((theta_index < 0), inc, noinc, theta_index)

    theta_indices = jnp.linspace(
        start=theta_index,
        stop=theta_index + theta_index_increment * num_beams,
        num=num_beams,
        endpoint=True,
        dtype=int,
    )[:, None]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(theta_arr[theta_indices], scan, s=2)
    ax.scatter(theta_arr[theta_indices], ref_scan, s=2)
    plt.show()

    # assert jnp.allclose(scan, ref_scan, atol=1e-01)
