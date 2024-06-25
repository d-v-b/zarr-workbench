from typing import Literal
import pytest
import os
import time
from v3_sharding.array_write import array_write
import numpy as np

def shape_to_mb(shape: tuple[int, ...], dtype: str | np.dtype):
    """
    Represent the number of elements in an array as a quantity of megabytes (MB)
    """
    return (np.prod(shape) * np.dtype(dtype).itemsize) / (1 << 20)

@pytest.fixture(scope='session')
def report_path(request: pytest.FixtureRequest) -> str:
    path = 'report.csv'
    with open('report.csv', mode='w') as fh:
        fh.write('engine,shard size (MB),chunk size (MB),throughput (MB/s)\n')
    return path

@pytest.mark.parametrize(
    'read_chunk_shape', 
    [
        (32,) * 3, 
        (64,) * 3, 
        (128,) * 3
        ])
@pytest.mark.parametrize('write_chunk_shape', [
    (64,) * 3,
    (128,) * 3,
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
    array_size_GB = shape_to_mb(old.shape, old.dtype)
    write_chunk_size_GB = shape_to_mb(write_chunk_shape, old.dtype)
    read_chunk_size_GB = shape_to_mb(read_chunk_shape, old.dtype)

    start = time.time()
    array_write(
        old, 
        out_chunks=write_chunk_shape, 
        shard_chunks=read_chunk_shape, 
        dest_url=os.path.join(str(tmpdir), 'test.zarr', 'array'),
        engine=engine)

    elapsed = time.time() - start
    throughput = array_size_GB / elapsed

    with open(report_path, mode='a') as fh:
        fh.write(f'{engine},{write_chunk_size_GB},{read_chunk_size_GB}, {throughput}\n')
