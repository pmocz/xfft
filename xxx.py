from enum import Enum

import jax
from jax.sharding import PartitionSpec


class Dist(Enum):
    """Describes a SLAB data decomposition

    For a X*Y*Z array, SLABS_X indicates the array is
    distributed along the first dimension, i.e., each
    device owns a slab of size (X // nGPUs)*Y*Z
    SLABS_Y indicates the array is distributed along the
    second dimension, with each device owning a slab
    of size X*(Y // nGPUs)*Z.
    """

    SLABS_X = "SLABS_X"
    SLABS_Y = "SLABS_Y"

    @staticmethod
    def create(string):
        if string == "X":
            return Dist.SLABS_X
        elif string == "Y":
            return Dist.SLABS_Y
        else:
            raise RuntimeError("Wrong dist")

    @property
    def opposite(dist):
        if dist == Dist.SLABS_X:
            return Dist.SLABS_Y
        else:
            return Dist.SLABS_X

    @property
    def _C_enum(dist):
        if dist == Dist.SLABS_X:
            return 0
        else:
            return 1

    def fft_axes(self, fft_rank):
        if self == Dist.SLABS_X:
            return list(range(1, fft_rank))
        else:
            return [0]

    def slab_shape(self, fft_dims):
        ngpus = jax.device_count()
        if self == Dist.SLABS_X:
            return (fft_dims[0] // ngpus, fft_dims[1], *fft_dims[2:])
        else:
            return (fft_dims[0], fft_dims[1] // ngpus, *fft_dims[2:])

    def fft_shape(self, local_shape):
        ngpus = jax.device_count()
        if self == Dist.SLABS_X:
            return (local_shape[0] * ngpus, local_shape[1], *local_shape[2:])
        else:
            return (local_shape[0], local_shape[1] * ngpus, *local_shape[2:])

    @property
    def part_spec(dist):
        if dist == Dist.SLABS_X:
            return PartitionSpec("gpus", None)
        else:
            return PartitionSpec(None, "gpus")


class Dir(Enum):
    """Describe the FFT direction

    FWD is the forward, unnormalized, direction.
    BWD is the backward, normalized by 1/N, direction,
    with N the product of the dimensions.
    """

    FWD = "FWD"
    INV = "INV"

    @property
    def _C_enum(dir):
        if dir == Dir.FWD:
            return 0
        else:
            return 1

    @property
    def opposite(dir):
        if dir == Dir.FWD:
            return Dir.INV
        else:
            return Dir.FWD


from functools import partial

import jax
from jax.sharding import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning
# from fft_common import Dir


def _fft(x, dist, dir):
    """Compute a local FFT along the appropriate axes (based on dist), in the
    forward or backward direction"""

    if dir == Dir.FWD:
        return jax.numpy.fft.fftn(x, axes=dist.fft_axes(len(x.shape)))
    else:
        return jax.numpy.fft.ifftn(x, axes=dist.fft_axes(len(x.shape)))


def _supported_sharding(sharding, dist):
    return NamedSharding(sharding.mesh, dist.part_spec)


def _partition(mesh, arg_shapes, result_shape, dist, dir):
    arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
    return (
        mesh,
        lambda x: _fft(x, dist, dir),
        _supported_sharding(arg_shardings[0], dist),
        (_supported_sharding(arg_shardings[0], dist),),
    )


def _infer_sharding_from_operands(mesh, arg_shapes, result_shape, dist, dir):
    arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
    return _supported_sharding(arg_shardings[0], dist)


def fft(x, dist, dir):
    """Extends jax.numpy.fft.fftn to support sharding along the first or
    second direction, without intermediate re-sharding"""

    @custom_partitioning
    def _fft_(x):
        return _fft(x, dist, dir)

    _fft_.def_partition(
        infer_sharding_from_operands=partial(
            _infer_sharding_from_operands, dist=dist, dir=dir
        ),
        partition=partial(_partition, dist=dist, dir=dir),
        sharding_rule="i j k -> i j k",
    )

    return _fft_(x)


def xfft(x, dist, dir):
    """Compute the discrete Fourier transform using a JAX-only implementation.

    Arguments:
    x    -- the input tensor
    dist -- the data decomposition of x.
            Should be an instance of fft_common.Dist
    dir  -- the direction of the transform.
            Should be an instance of fft_common.Dir

    Returns the transformed tensor.
    The output tensoris distributed according to dist.opposite

    This function should be used with jit like

        jit(xfft,
            in_shardings=sharding,
            out_shardings=sharding_opposite,
            static_argnums=[1, 2]
            )(x, dist, dir)
    """

    # If dist == Dist.SLABS_X, FFT along Y and Z
    x = fft(x, dist, dir)

    # Implicitly re-shards to match the required
    # input sharding of the next fft(..., dist.opposite, ...)

    # If dist == Dist.SLABS_X, FFT along X
    x = fft(x, dist.opposite, dir)

    return x
