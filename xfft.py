import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import argparse
import time
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Philip Mocz (2025)
# Flatiron Institute

# Distributed 3D fft using JAX
# TODO: turn this into a proper mini-library


def main():
    parser = argparse.ArgumentParser(description="Distributed 3D FFT using JAX")
    parser.add_argument("--res", type=int, required=True, help="Resolution of the grid")
    parser.add_argument(
        "--double", action="store_true", help="Use double precision for calculations"
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed FFT"
    )
    args = parser.parse_args()

    N = args.res

    if args.distributed:
        jax.distributed.initialize()

    # Create mesh and sharding for distributed computation
    n_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(devices, axis_names=("gpus",))
    sharding = NamedSharding(mesh, PartitionSpec(None, "gpus"))

    if jax.process_index() == 0:
        print(f"Calculating FFT with resolution {N} on {n_devices} devices")

    if args.double:
        precision = "double"
        jax.config.update("jax_enable_x64", True)
    else:
        precision = "single"
        jax.config.update("jax_enable_x64", False)

    N_trials = 30

    # Create a test array
    L = 2.0 * jnp.pi
    xlin = jnp.linspace(0, L, num=N + 1)
    xlin = xlin[0:N]
    xx, yy, zz = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    # Apply sharding to meshgrid arrays
    xx = jax.lax.with_sharding_constraint(xx, sharding)
    yy = jax.lax.with_sharding_constraint(yy, sharding)
    zz = jax.lax.with_sharding_constraint(zz, sharding)

    vx = jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz)
    del xx, yy, zz  # free memory

    # Perform the FFT
    # warm-up
    vx_hat = jfft.fftn(vx)
    vx_hat.block_until_ready()
    # time it
    start_time = time.time()
    for i in range(N_trials):
        vx_hat = jfft.fftn(vx)
        vx_hat.block_until_ready()
    end_time = time.time()

    timing = 1000.0 * (end_time - start_time) / N_trials

    if jax.process_index() == 0:
        print(f"FFT computed in {timing:.6f} ms")

        log_filename = f"log_jfft_n{N}_d{n_devices}_{precision}.txt"
        with open(log_filename, "w") as log_file:
            log_file.write(f"{timing:.6f}\n")


if __name__ == "__main__":
    main()
