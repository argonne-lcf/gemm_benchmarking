#!/bin/bash
set -euo pipefail

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=12

echo "Jobid: $PBS_JOBID"
echo "NUM_OF_NODES=${NNODES}  RANKS_PER_NODE=${RANKS_PER_NODE}"
module list

mpicxx -o gemm gemm.cpp -fsycl -DMKL_ILP64 -fiopenmp -qmkl=parallel -DSAVE

RESULTS=results
rm -rf "$RESULTS"
mkdir -p "$RESULTS"
cd "$RESULTS"

GPU_CPU_BIND=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100

OMP_NUM_THREADS=8 mpirun \
    -n $(( NNODES * 12 )) -ppn 12 --cpu-bind "$GPU_CPU_BIND" \
    gpu_tile_compact.sh ../gemm gpu

mv data.txt data_gpu.txt

OMP_NUM_THREADS=51 mpirun \
    -n $(( NNODES * 2 )) -ppn 2 --cpu-bind=list:1-51:53-103 \
    ../set_hbm.sh ../gemm cpu

mv data.txt data_cpu.txt

cat data_gpu.txt data_cpu.txt > data.txt

export OMP_NUM_THREADS=1
uv run python ../mine_paper.py -f data.txt --no-post-process > output_no_post_process
mv data.png output_no_post_process.png
uv run python ../mine_paper.py -f data.txt > output_post_process
mv data.png output_post_process.png
