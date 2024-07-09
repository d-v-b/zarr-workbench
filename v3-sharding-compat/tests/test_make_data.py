import pytest
from pathlib import Path
import os
import zarr
import numpy as np
from v3_sharding_compat import normalize_chunking, zcreate

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
    print(shape, chunking, res)
    assert np.prod(np.divide(shape, res)) == chunking
    
        

@pytest.mark.parametrize('zarr_format', [2, 3])
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
def test_create_array(data_dir, zarr_format, codecs, shape, chunking, chunk_key, dtype):
    attributes = {'about': 'a test array'}
    store = zarr.store.LocalStore(root=data_dir, mode='w')    
    outer_chunks_normed = normalize_chunking(shape, chunking[0])
    inner_chunks_normed = normalize_chunking(outer_chunks_normed, chunking[1])
    chunk_key_translated = "dot" if chunk_key is "." else "slash"
    
    if chunking[1] > 1:
        if zarr_format == 2:
            pytest.xfail()
            raise NotImplementedError
        chunks = {
            'shape': outer_chunks_normed, 
            'chunks': {'shape': inner_chunks_normed}}
    
    else:
        chunks = outer_chunks_normed
    
    path = f'zarr-{zarr_format}/dtype-{dtype}/nd-{len(shape)}/co-{chunking[0]}_ci-{chunking[1]}_ck-{chunk_key_translated}'
    
    array = zcreate(
        store=store,
        path=path,
        shape=shape, 
        dtype=dtype, 
        zarr_format=zarr_format, 
        attributes=attributes,
        chunks=chunks,
        dimension_separator=chunk_key
        )

    array[:] = np.arange(np.prod(shape)).reshape(shape).astype(dtype)