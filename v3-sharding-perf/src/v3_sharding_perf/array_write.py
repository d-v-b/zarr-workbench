from typing import Any, Literal
from zarr.store import LocalStore, RemoteStore
from zarr.codecs import ShardingCodec, BloscCodec, BytesCodec
import zarr
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tensorstore as ts
import os

async def alist(it):
    out = []
    async for a in it:
        out.append(a)
    return out

def zarray_to_tsarray(zarray: zarr.Array) -> Any:
    if isinstance(zarray.store_path.store, LocalStore):
        return ts.open({
        'driver': 'zarr3',
            'kvstore': {
                'driver': 'file',
                'path': os.path.join(zarray.store_path.store.root, zarray.store_path.path),
            },
            }).result()
    else:
        raise ValueError('Only localstore-backed zarr arrays for now.')


def set(old, new, region):
    new[region] = old[region]
    return region

def slices_from_chunks(shape: tuple[int, ...], chunks: tuple[int, ...]):
    chunk_grid_indices = product(*(range(int(np.ceil(s / c))) for s,c in zip(shape, chunks)))
    chunk_grid_slices = tuple(tuple(slice(o * c, o * c + c) for o, c in zip(cindex, chunks)) for cindex in chunk_grid_indices)
    return chunk_grid_slices

def copy_array_serial_zarr(old, new: zarr.Array, chunk_slices: tuple[slice, ...]):
    for chunk_slice in chunk_slices:
        new[chunk_slice] = old[chunk_slice]

def copy_array_serial_tensorstore(old: np.ndarray, new: zarr.Array, chunk_slices: tuple[slice, ...]):
    ts_array = zarray_to_tsarray(new)
    for chunk_slice in chunk_slices:
        ts_array[chunk_slice] = old[chunk_slice]

def array_write(
        old: np.ndarray, 
        out_chunks: tuple[int, ...], 
        shard_chunks: tuple[int, ...], 
        dest_url: str, 
        engine: Literal["zarr-python", "tensorstore"],
        strategy: Literal["bulk", "per_chunk"]):

    codec_pipeline = ShardingCodec(chunk_shape=shard_chunks, codecs=(BytesCodec(), BloscCodec(cname="zstd")))
    pre, suffix, write_array_path = dest_url.partition('.zarr/')
    write_path = pre+suffix
    write_store = LocalStore(root=write_path, mode='w')
    new = zarr.create(
        store=write_store,
        shape=old.shape,
        dtype=old.dtype,
        path=write_array_path,
        zarr_format=3,
        chunk_shape=out_chunks,
        codecs=(codec_pipeline,), 
        overwrite=True)

    chunk_slices = slices_from_chunks(new.shape, new.chunks)
    
    if engine == 'zarr-python':
        if strategy == 'per_chunk':
            copy_array_serial_zarr(old, new, chunk_slices)
        else:
            copy_array_bulk_zarr(old, new)
    else:
        if strategy == 'per_chunk':
            copy_array_serial_tensorstore(old, new, chunk_slices)
        else:
            copy_array_serial_tensorstore(old, new, (slice(None,), ))
    
    return new
