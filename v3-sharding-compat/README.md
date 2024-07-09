# v3-sharding-compat

A collection of test arrays.

Warning: anything about this project could change at any time. 

## making data
1. clone the parent repo
2. `cd v3 sharding compat`
3. `hatch run pytest tests`

## data layout
The arrays are stored in the `data` directory.

The arrays are parametrized over the following properties:
- the zarr version (2 or 3)
- data type
- dimensionality
- outer chunks, i.e. the number of files in the array.
- inner chunks, i.e. the number of partitions within each outer chunk.
- dimension separator ("." or "/")

The above properties are encoded in the path of the arrays. 
E.g., [`zarr-2/dtype-bool/nd-1/co-64_ci-1_ck-dot`](https://github.com/d-v-b/zarr-workbench/tree/main/v3-sharding-compat/data/zarr-2/dtype-bool/nd-1/co-64_ci-1_ck-dot) is a zarr v2 array with the following properties:
- zarr format: 2 (encoded in the path as `zarr-2`)
- dtype: `bool` (encoded in the path as `dtype-bool`)
- dimensionality: 1 (encoded in the path as `nd-1`)
- number of outer chunks: 64 (encoded in the path as `co-64`)
- number of inner chunks: 1 (encoded in the path as `ci-1`)
- dimension separator: "." (encoded in the path as `ck-dot`)

All v2 arrays will have 1 inner chunk, because v2 does not support inner chunking (i.e., sharding).

[`zarr-3/dtype-int8/nd-3/co-8_ci-8_ck-slash`](https://github.com/d-v-b/zarr-workbench/tree/main/v3-sharding-compat/data/zarr-3/dtype-int8/nd-3/co-8_ci-8_ck-slash) describes a zarr v3 array with the following properties:
- zarr format: 3 (encoded in the path as `zarr-3`)
- dtype: `int8` (encoded in the path as `dtype-int8`)
- dimensionality: 3 (encoded in the path as `nd-3`)
- number of outer chunks: 8 (encoded in the path as `co-8`)
- number of inner chunks: 8 (encoded in the path as `ci-8`)
- dimension separator: "/" (encoded in the path as `ck-slash`)
