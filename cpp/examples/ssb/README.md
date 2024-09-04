## SSB examples

These examples are to benchmark performance of cuCollection library for ssb queries. The queries are implemented in a very similar way that CRYSTAL: https://github.com/anilshanbhag/crystal has done by pipelining the operators in data centric way.

### Generate data

Clone the above mentioned crystal repo.
Follow the steps to generate ssb data. 

Change the BASE_PATH in ssb_utils.h

#### Buffer overflow issue during test data generation

Resolution: https://github.com/electrum/ssb-dbgen/issues/6 
Change the MAXAGG_LEN at line 120 in file test/ssb/dbgen/shared.h to 32. 



### Build the query

Within the cudf_dev conda environment, following standard cmake procedure can be followed
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug .. 
cmake --build .
```

### Running the query

Launch the command

```
./q21-cuco
```