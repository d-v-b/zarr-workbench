# SPDX-FileCopyrightText: 2024-present Davis Vann Bennett <davis.v.bennett@gmail.com>
#
# SPDX-License-Identifier: MIT
import numpy as np
import zarr 

def normalize_chunking(shape: tuple[int, ...], num_chunks: int):
    """
    Convert an implicit chunking specification to a full chunking spec
    """
    num_chunks_remaining: int = num_chunks
    chunks_normalized = np.array(shape, dtype='int')[::-1]
    
    for idx, s in enumerate(reversed(shape)):
        
        if num_chunks_remaining == 1:
            break
        
        gcd: int = np.gcd(num_chunks_remaining, s)
        
        if gcd == 1:
            pass
        
        chunks_normalized[idx] = chunks_normalized[idx] // gcd
        num_chunks_remaining  = num_chunks_remaining // gcd
    assert np.prod(np.divide(shape, chunks_normalized)) == num_chunks
    res = chunks_normalized[::-1]
    return tuple(res.tolist())

def zcreate(
        store,
        path: str, 
        shape, 
        dtype, 
        zarr_format, 
        attributes, 
        chunks,
        dimension_separator = '/',
        compression = (),
        filters = (),
        sharding_codec="auto",
        chunk_layout = 'default'):
    
    if isinstance(chunks, tuple):
        shard_size = chunks
        subshard_size = None

    else:
        shard_size = chunks['shape']

        if 'chunks' in chunks:
            subshard_size = chunks['chunks']['shape']
        else:
            subshard_size = None

    if subshard_size is not None:
        if sharding_codec == "auto":
            sharding_codec = zarr.codecs.ShardingCodec
        else:
            sharding_codec = sharding_codec
        
        codecs = (*filters, sharding_codec(chunk_shape=subshard_size, codecs=(zarr.codecs.BytesCodec(), *compression)))
    else:
        codecs = (*filters, zarr.codecs.BytesCodec(), *compression)

    if zarr_format == 2:
        if compression == ():
            compressor = None
        else:
            compressor = compression[0]

        return zarr.create(
            store=store,
            path=path,
            shape=shape, 
            dtype=dtype, 
            zarr_format=zarr_format, 
            attributes=attributes,
            chunks=shard_size,
            filters=filters,
            compressor=compressor,
            overwrite=True,
            )
    else:
        return zarr.create(
            store=store,
            path=path,
            shape=shape, 
            dtype=dtype, 
            zarr_format=zarr_format, 
            attributes=attributes,
            chunk_shape=shard_size,
            chunk_key_encoding=(chunk_layout, dimension_separator),
            codecs = codecs,
            overwrite=True,
            )