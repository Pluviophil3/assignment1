#!/bin/bash
# Submit one configuration with submit_heat_mpi_benchmark.sh <n> <p>.
#SBATCH --job-name=heat_mpi_bench
#SBATCH --output=slurm_heat_mpi_%j.out
#SBATCH --nodes=1
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <300|400|500> <1|2|4|8|16|32>" >&2
    exit 2
fi
n=$1
p=$2
case "$n" in 300|400|500) ;; *) exit 2 ;; esac
case "$p" in 1|2|4|8|16|32) ;; *) exit 2 ;; esac
if [[ "${SLURM_NTASKS:?}" -ne "$p" ]]; then
    echo "Slurm allocated $SLURM_NTASKS tasks; expected $p" >&2
    exit 2
fi

mkdir -p bench_results
mpi_bin="bench_results/heat_mpi_${SLURM_JOB_ID}"
seq_bin="bench_results/heat_seq_${SLURM_JOB_ID}"
mpicxx -O3 heat_mpi.cpp -o "$mpi_bin"
g++ -O3 heat_seq.cpp -o "$seq_bin"

# A sequential output on the same node checks every MPI run.
seq_solution="bench_results/seq_check_n${n}_p${p}.txt"
"$seq_bin" "$n" "$n" 100000 "$seq_solution" \
    > "bench_results/seq_check_n${n}_p${p}.log"

csv="bench_results/mpi_n${n}_p${p}.csv"
solution="bench_results/mpi_solution_n${n}_p${p}.txt"
printf 'n,p,run,seconds\n' > "$csv"
for run in 1 2 3 4 5; do
    log="bench_results/mpi_n${n}_p${p}_run${run}.log"
    mpiexec -n "$p" "$mpi_bin" "$n" "$n" 100000 "$solution" > "$log"
    cmp "$seq_solution" "$solution"
    seconds=$(awk '/^Elapsed time = / {print $4}' "$log")
    if [[ ! "$seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Could not read elapsed time from $log" >&2
        exit 1
    fi
    printf '%s,%s,%s,%s\n' "$n" "$p" "$run" "$seconds" >> "$csv"
    grep -E '^(Converged|Not converged) after ' "$log"
    echo "mpi: n=${n}, p=${p}, run=${run}, seconds=${seconds}"
done
echo "PASS: n=${n}, p=${p}; all five results match sequential"
echo "Saved $csv"
