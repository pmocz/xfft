import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import argparse
import time
from jax.experimental import mesh_utils
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from typing import Callable

# Philip Mocz (2025)
# Flatiron Institute

# Distributed 3D fft using JAX
# TODO: turn this into a proper mini-library


def fft_partitioner(
    fft_func: Callable[[jax.Array], jax.Array], partition_spec: PartitionSpec
):
    @custom_partitioning
    def func(x):
        return fft_func(x)

    def supported_sharding(sharding, shape):
        return NamedSharding(sharding.mesh, partition_spec)

    def partition(arg_shapes, arg_shardings, result_shape, result_sharding):
        return (
            fft_func,
            supported_sharding(arg_shardings[0], arg_shapes[0]),
            (supported_sharding(arg_shardings[0], arg_shapes[0]),),
        )

    def infer_sharding_from_operands(arg_shapes, arg_shardings, shape):
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    func.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands, partition=partition
    )
    return func


def _fft_XY(x):
    return jax.numpy.fft.fftn(x, axes=[0, 1])


def _fft_Z(x):
    return jax.numpy.fft.fft(x, axis=2)


def _ifft_XY(x):
    return jax.numpy.fft.ifftn(x, axes=[0, 1])


def _ifft_Z(x):
    return jax.numpy.fft.ifft(x, axis=2)


fft_XY = fft_partitioner(_fft_XY, PartitionSpec(None, None, "gpus"))
fft_Z = fft_partitioner(_fft_Z, PartitionSpec(None, "gpus"))
ifft_XY = fft_partitioner(_ifft_XY, PartitionSpec(None, None, "gpus"))
ifft_Z = fft_partitioner(_ifft_Z, PartitionSpec(None, "gpus"))


def xfftn(x):
    x = fft_Z(x)
    x = fft_XY(x)
    return x


def ixfftn(x):
    x = ifft_XY(x)
    x = ifft_Z(x)
    return x


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

    # set up xfft
    xfft = jax.jit(
        xfftn,
        donate_argnums=0,  # doesn't help
        in_shardings=(NamedSharding(mesh, PartitionSpec(None, "gpus"))),
        out_shardings=(NamedSharding(mesh, PartitionSpec(None, "gpus"))),
    )

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
        print(f"JFFT computed in {timing:.6f} ms")

        log_filename = f"log_jfft_n{N}_d{n_devices}_{precision}.txt"
        with open(log_filename, "w") as log_file:
            log_file.write(f"{timing:.6f}\n")

    # Now do xfft
    # warm-up
    vx_hat = xfft(vx)
    vx_hat.block_until_ready()
    # time it
    start_time = time.time()
    for i in range(N_trials):
        vx_hat = xfft(vx)
        vx_hat.block_until_ready()
    end_time = time.time()

    timing = 1000.0 * (end_time - start_time) / N_trials

    if jax.process_index() == 0:
        print(f"XFFT computed in {timing:.6f} ms")

        log_filename = f"log_xfft_n{N}_d{n_devices}_{precision}.txt"
        with open(log_filename, "w") as log_file:
            log_file.write(f"{timing:.6f}\n")


if __name__ == "__main__":
    main()
