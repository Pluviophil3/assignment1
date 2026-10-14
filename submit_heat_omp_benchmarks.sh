#!/bin/bash
# Compile once, then submit one Slurm job for each required problem size.
set -euo pipefail
cd "$(dirname "$0")"

g++ -O3 heat_seq.cpp -o heat_seq
g++ -O3 -fopenmp heat_omp.cpp -o heat_omp

for n in 300 400 500; do
    sbatch -p cmsc5702_hpc -q cmsc5702 \
        --export="ALL,HEAT_N=${n}" bench_heat_omp.sh
done
