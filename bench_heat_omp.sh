#!/bin/bash
# Run one problem size on a single allocated node.
# Submit with: sbatch -p cmsc5702_hpc -q cmsc5702 --export=ALL,HEAT_N=300 bench_heat_omp.sh
#SBATCH --job-name=heat_omp_bench
#SBATCH --output=slurm_heat_omp_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

case "${HEAT_N:-}" in
    300|400|500) ;;
    *) echo "HEAT_N must be 300, 400, or 500" >&2; exit 2 ;;
esac

if [[ ! -x ./heat_seq || ! -x ./heat_omp ]]; then
    echo "Compile heat_seq and heat_omp before submitting this job" >&2
    exit 2
fi

mkdir -p bench_results
seq_csv="bench_results/seq_n${HEAT_N}.csv"
omp_csv="bench_results/omp_n${HEAT_N}.csv"
printf 'n,p,run,seconds\n' > "$seq_csv"
printf 'n,p,run,seconds\n' > "$omp_csv"

export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

run_five() {
    local program="$1" p="$2" csv="$3" run log solution seconds
    solution="bench_results/${program}_n${HEAT_N}_p${p}.txt"
    for run in 1 2 3 4 5; do
        log="bench_results/${program}_n${HEAT_N}_p${p}_run${run}.log"
        export OMP_NUM_THREADS="$p"
        "./heat_${program}" "$HEAT_N" "$HEAT_N" 100000 "$solution" > "$log"
        seconds=$(awk '/^Elapsed time = / {print $4}' "$log")
        if [[ ! "$seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "Could not read elapsed time from $log" >&2
            exit 1
        fi
        printf '%s,%s,%s,%s\n' "$HEAT_N" "$p" "$run" "$seconds" >> "$csv"
        grep -E '^(Converged|Not converged) after ' "$log"
        echo "${program}: n=${HEAT_N}, p=${p}, run=${run}, seconds=${seconds}"
    done
}

run_five seq 1 "$seq_csv"
for p in 1 2 4 8 16 32; do
    run_five omp "$p" "$omp_csv"
    cmp "bench_results/seq_n${HEAT_N}_p1.txt" \
        "bench_results/omp_n${HEAT_N}_p${p}.txt"
done

echo "PASS: n=${HEAT_N}; all OpenMP results match the sequential result"
echo "Saved $seq_csv and $omp_csv"
