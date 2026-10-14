#!/bin/bash
# Compile and compare the MPI solver with the sequential solver on a compute node.
#SBATCH --job-name=heat_mpi_check
#SBATCH --output=mpi_validate_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:10:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"

mpi_compiler=$(command -v mpicxx)
mpi_runner="$(dirname "$mpi_compiler")/mpiexec"
if [[ ! -x "$mpi_runner" ]]; then
    echo "Matching mpiexec was not found beside $mpi_compiler" >&2
    exit 1
fi
echo "Compiler: $mpi_compiler"
echo "Runner: $mpi_runner"

"$mpi_compiler" -O3 heat_mpi.cpp -o heat_mpi
g++ -O3 heat_seq.cpp -o heat_seq

mkdir -p mpi_validation
./heat_seq 20 20 100000 mpi_validation/seq20.txt > mpi_validation/seq20.log

for p in 1 2 4; do
    "$mpi_runner" -n "$p" ./heat_mpi 20 20 100000 \
        "mpi_validation/mpi20_p${p}.txt" > "mpi_validation/mpi20_p${p}.log"
    cmp mpi_validation/seq20.txt "mpi_validation/mpi20_p${p}.txt"
    grep -E '^(Converged|Not converged) after ' "mpi_validation/mpi20_p${p}.log"
    echo "PASS: p=${p} temperature output matches sequential"
done
