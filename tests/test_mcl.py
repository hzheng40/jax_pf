from jax_pf.src.mcl import compute_sensor_model

import pytest
import numpy as np
import jax
import jax.numpy as jnp


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
