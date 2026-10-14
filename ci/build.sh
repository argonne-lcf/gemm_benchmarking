#!/bin/bash
set -euo pipefail

mpicxx -o gemm gemm.cpp -fsycl -DMKL_ILP64 -fiopenmp -qmkl=parallel -DSAVE
