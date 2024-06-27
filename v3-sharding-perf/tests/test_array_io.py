from typing import Literal
import pytest
import os
import time
from v3_sharding_perf.array_write import array_write, zarray_to_tsarray
import numpy as np
from shutil import rmtree
from pathlib import Path
from glob import glob
REPORT_DIR = Path(os.path.realpath(__file__)).parent.parent / 'reports'

def shape_to_mb(shape: tuple[int, ...], dtype: str | np.dtype):
    """
    Represent the number of elements in an array as a quantity of megabytes (MB)
    """
    return (np.prod(shape) * np.dtype(dtype).itemsize) / (1 << 20)

@pytest.fixture(scope='session')
def report_dir():
    """
    Delete report files, if they exist
    """
    for p in REPORT_DIR.glob('*.csv'):
        os.remove(p) 

@pytest.fixture(scope='function')
def report_path(report_dir, request: pytest.FixtureRequest) -> str:
    path = os.path.join(REPORT_DIR, f'{request.function.__name__}.csv')
    if not os.path.exists(path):
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, mode='w') as fh:
            fh.write('engine,shard size (MB),chunk size (MB),throughput (MB/s)\n')
    
    return path

@pytest.mark.parametrize(
    'read_chunk_shape', 
    [
        (64,) * 3, 
        (128,) * 3
        ])
@pytest.mark.parametrize('write_chunk_shape', [
    (256,) * 3,
    (512,) * 3,
    (1024,) * 3,])
@pytest.mark.parametrize('engine', [
    'zarr-python', 
    'tensorstore'
    ])
def test_array_write(
    tmpdir, 
    report_path: str,
    engine: Literal["zarr-python", "tensorstore"],
    write_chunk_shape: tuple[int, ...], 
    read_chunk_shape: tuple[int, ...]):
    
    num_chunks = 2  
    array_shape = (write_chunk_shape[0] * num_chunks, *write_chunk_shape[1:])
    dtype='uint8'

    old = np.arange(np.prod(array_shape), dtype=dtype).reshape(array_shape)
    array_size_MB = shape_to_mb(old.shape, old.dtype)
    write_chunk_size_MB = shape_to_mb(write_chunk_shape, old.dtype)
    read_chunk_size_MB = shape_to_mb(read_chunk_shape, old.dtype)

    start = time.time()
    array_write(
        old, 
        out_chunks=write_chunk_shape, 
        shard_chunks=read_chunk_shape, 
        dest_url=os.path.join(str(tmpdir), 'test.zarr', 'array'),
        engine=engine)

    elapsed = time.time() - start
    throughput = array_size_MB / elapsed

    with open(report_path, mode='a') as fh:
        fh.write(f'{engine},{write_chunk_size_MB},{read_chunk_size_MB}, {throughput}\n')

@pytest.mark.parametrize(
    'read_chunk_shape', 
    [
        (64,) * 3,
        (128,) * 3
        ])
@pytest.mark.parametrize('write_chunk_shape', [
    (256,) * 3,
    (512,) * 3,
    (1024,) * 3,])
@pytest.mark.parametrize('engine', [
    'zarr-python', 
    'tensorstore'
    ])
def test_array_read(
    tmpdir, 
    report_path: str,
    engine: Literal["zarr-python", "tensorstore"],
    write_chunk_shape: tuple[int, ...], 
    read_chunk_shape: tuple[int, ...]):
    
    num_chunks = 2  
    array_shape = (write_chunk_shape[0] * num_chunks, *write_chunk_shape[1:])
    dtype='uint8'

    old = np.arange(np.prod(array_shape), dtype=dtype).reshape(array_shape)
    array_size_MB = shape_to_mb(old.shape, old.dtype)
    write_chunk_size_MB = shape_to_mb(write_chunk_shape, old.dtype)
    read_chunk_size_MB = shape_to_mb(read_chunk_shape, old.dtype)

    arr = array_write(
        old, 
        out_chunks=write_chunk_shape, 
        shard_chunks=read_chunk_shape, 
        dest_url=os.path.join(str(tmpdir), 'test.zarr', 'array'),
        engine='tensorstore')

    del old

    # read the array back
    start = time.time()
    if engine == 'zarr-python':
        result = arr[:]
    else:
        result = zarray_to_tsarray(arr)[:].read().result()

    elapsed = time.time() - start
    throughput = array_size_MB / elapsed

    with open(report_path, mode='a') as fh:
        fh.write(f'{engine},{write_chunk_size_MB},{read_chunk_size_MB}, {throughput}\n')
