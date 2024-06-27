import pytest
from pathlib import Path
import os
import zarr
import numpy as np
from v3_sharding_compat import normalize_chunking

REPORT_DIR = Path(os.path.realpath(__file__)).parent.parent / 'data'

@pytest.fixture(scope='session')
def data_dir():
    return REPORT_DIR


@pytest.mark.parametrize('shape', [
    (512,), 
    (8, 64), 
    (8,8,8), 
    (2,4,8,8), 
    (2,2,2,8,8), 
    (2,2,2,2,4,8)])
@pytest.mark.parametrize(
    'chunking', [
    1,
    8,
    64
    ])
def test_normalize_chunking(shape, chunking):
    res = normalize_chunking(shape, chunking)
    assert np.prod(np.divide(shape, res)) == chunking[0]
    
        

@pytest.mark.skip
@pytest.mark.parametrize('zarr_format', [2,3])
@pytest.mark.parametrize('codecs', (None,))
@pytest.mark.parametrize('shape', [
    (512,), 
    (8, 64), 
    (8,8,8), 
    (2,4,8,8), 
    (2,2,2,8,8), 
    (2,2,2,2,4,8)])
@pytest.mark.parametrize(
    'chunking', [
    (1,64),
    (8,8),
    (64,1)
    ])
@pytest.mark.parametrize('chunk_key', ['.','/'])
@pytest.mark.parametrize('dtype', [
    'bool', 
    'int8', 
    'uint8', 
    'int16', 
    'uint16', 
    'int32', 
    'uint32', 
    'int64', 
    'uint64', 
    'float32', 
    'float64'])
def create_array(data_dir, zarr_format, codecs, shape, chunking, chunk_key, dtype):
    attributes = {'about': 'a test array'}
    store = zarr.store.LocalStore(root=data_dir)
    
    chunks_actual = normalize_chunking(shape, chunking[0])

    array = zarr.create(
        store=store,
        shape=shape, 
        dtype=dtype, 
        zarr_format=zarr_format, 
        attributes=attributes,
        chunk_shape=chunks_actual,
        chunk_key_encoding=('default', chunk_key),
        codecs=codecs,
        )