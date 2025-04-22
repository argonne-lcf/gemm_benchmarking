# gemm_benchmarking

## Compiling:
```
> make
mpicxx -o gemm gemm.cpp -fsycl -DMKL_ILP64 -fiopenmp -qmkl=parallel
```

## Running:

We do two runs to get the GPU and CPU flop-rates. Note that you can runs on multiple nodes and you can get the "Best" single-stack or single-GPU (pair of stacks) value from the "Subset of Rank" value

```
# Bench CPU: Best One Socket of Xeon (and one GPU stack used for verification)
OMP_NUM_THREADS=51 mpirun -n $(( $(wc -l < $PBS_NODEFILE) * 2)) -ppn 2 --cpu-bind=list:1-51:53-103 ./set_hbm.sh ./gemm cpu

# Bench GPU: Bost of 2 Stacks of a single PVC (and 8 threads CPU used for verification)
OMP_NUM_THREADS=8 mpirun -n $(( $(wc -l < $PBS_NODEFILE) * 12)) -ppn 12 --cpu-bind list:1-8:9-16:17-24:25-32:33-40:41-48:52-59:60-67:68-75:76-83:84-91:92-99 gpu_tile_compact.sh ./gemm gpu
```

## Output

Q == Quartile 
```
Result For DGEMM (sample size: 6)
-Min 29400.5 GFlop/s
-Q1 29710.6 GFlop/s
-Q2(median) 29819.1 GFlop/s
-Q3 29955.7 GFlop/s
-Max 30119.5 GFlop/s
```


## Options

-DSAVE  Save all the `flops` off all rank in $Name.txt

-DITER_MAX Maximun number of Iteration (default 100)

-DITER_MIN Mimun number of Iteration  (default 10). The code will stop when ITER_MIN consecutive run doesn't produce a new best number
