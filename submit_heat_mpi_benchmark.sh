#!/bin/bash
# Submit a single MPI configuration to respect the two-job QoS limit.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <300|400|500> <1|2|4|8|16|32>" >&2
    exit 2
fi
case "$1" in 300|400|500) ;; *) echo "Invalid n" >&2; exit 2 ;; esac
case "$2" in 1|2|4|8|16|32) ;; *) echo "Invalid p" >&2; exit 2 ;; esac

sbatch -p cmsc5702_hpc -q cmsc5702 --export=NONE \
    --ntasks="$2" bench_heat_mpi.sh "$1" "$2"
