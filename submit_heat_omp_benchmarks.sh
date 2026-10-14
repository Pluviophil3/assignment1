#!/bin/bash
# Compile and submit one problem size at a time to respect the course QoS limit.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <300|400|500>" >&2
    exit 2
fi
case "$1" in
    300|400|500) ;;
    *) echo "Problem size must be 300, 400, or 500" >&2; exit 2 ;;
esac

g++ -O3 heat_seq.cpp -o heat_seq
g++ -O3 -fopenmp heat_omp.cpp -o heat_omp

sbatch -p cmsc5702_hpc -q cmsc5702 --time="${HEAT_TIME:-00:30:00}" \
    --export="ALL,HEAT_N=$1" bench_heat_omp.sh
