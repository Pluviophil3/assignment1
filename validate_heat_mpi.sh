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
mpi_runner=$(command -v mpiexec)
echo "Compiler: $mpi_compiler"
echo "Runner: $mpi_runner"

"$mpi_compiler" -O3 heat_mpi.cpp -o heat_mpi
g++ -O3 heat_seq.cpp -o heat_seq

mkdir -p mpi_validation
./heat_seq 20 20 100000 mpi_validation/seq20.txt > mpi_validation/seq20.log

for p in 1 2 4; do
    # Use the command name: Open MPI interprets an absolute /bin/mpiexec
    # as an installation prefix of / and rejects that empty prefix.
    mpiexec -n "$p" ./heat_mpi 20 20 100000 \
        "mpi_validation/mpi20_p${p}.txt" > "mpi_validation/mpi20_p${p}.log"
    cmp mpi_validation/seq20.txt "mpi_validation/mpi20_p${p}.txt"
    grep -E '^(Converged|Not converged) after ' "mpi_validation/mpi20_p${p}.log"
    echo "PASS: p=${p} temperature output matches sequential"
done

# Eleven rows cannot be divided evenly among four processes.
./heat_seq 11 13 100000 mpi_validation/seq11x13.txt > mpi_validation/seq11x13.log
mpiexec -n 4 ./heat_mpi 11 13 100000 mpi_validation/mpi11x13.txt \
    > mpi_validation/mpi11x13.log
cmp mpi_validation/seq11x13.txt mpi_validation/mpi11x13.txt
echo "PASS: uneven 11x13 decomposition matches sequential"

# Exercise the maximum-iteration stop without convergence.
./heat_seq 11 13 1 mpi_validation/seq11x13_limit.txt \
    > mpi_validation/seq11x13_limit.log
mpiexec -n 4 ./heat_mpi 11 13 1 mpi_validation/mpi11x13_limit.txt \
    > mpi_validation/mpi11x13_limit.log
cmp mpi_validation/seq11x13_limit.txt mpi_validation/mpi11x13_limit.txt
grep '^Not converged after 1 iterations' mpi_validation/mpi11x13_limit.log
echo "PASS: maximum-iteration stop matches sequential"
