## openfe_performance_benchmarks
Performance benchmarks for OpenFE Protocols


## How to use this

The benchmark simulation scripts are found under `benchmark`.

`md_benchmark.py`: this runs conventional MD using OpenMM (uses an underlying OpenFE Protocol for conventional MD).
`rbfe_benchmark.py`: this runs a hybrid topology RBFE using OpenMMTools + OpenMM with OpenFE.

You can run the scripts in the following manner::

   python rbfe_benchmark.py --input_file ../data/benchmark.json --output_file rbfe_benchmark.out


This will output a JSON serialized dictionary keyed by system names and values with the performance in ns/day.

You can see the notebook under `results` on how to analyze these results.


## License

Everything here is under an MIT license.
