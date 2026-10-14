/*
 * MPI Jacobi solver for the 2D steady-state heat distribution problem.
 * Compile: mpicxx -O3 heat_mpi.cpp -o heat_mpi
 * Execute: mpiexec -n <processes> ./heat_mpi [rows] [cols] [max_its] [filename]
 */

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const double EPSILON = 0.001;
const int VIEWABLE_SIZE = 20;

void allocate_2d_array(int r, int c, double ***a);
void initialize_array(int m, int n, double **u);
void initialize_local_array(int m, int n, int first_row, int local_rows, double **u);
void print_solution(const string& filename, int m, int n, double **u);
int find_steady_state(int m, int n, int max_its, int first_row, int local_rows,
                      int rank, int processes, double **u, double **w,
                      double **global_w, double &final_diff);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int m = 200;
    int n = 200;
    int max_its = 100000;
    string prog_path(argv[0]);
    size_t pos = prog_path.find_last_of("/\\");
    string prog_name = (pos != string::npos) ? prog_path.substr(pos + 1) : prog_path;

    if (argc > 1) {
        m = stoi(argv[1]);
        n = m;
    }
    if (argc > 2) n = stoi(argv[2]);
    if (argc > 3) max_its = stoi(argv[3]);

    string filename = prog_name + "-" + to_string(m) + "x" + to_string(n) + ".txt";
    if (argc > 4) filename = argv[4];
    if (argc > 5 || m < 2 || n < 2 || max_its < 1 || processes > m) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0]
                 << " [rows] [cols] [max_its] [filename]\n"
                 << "Require rows and cols >= 2, max_its >= 1, and processes <= rows.\n";
        }
        MPI_Finalize();
        return -1;
    }

    // Assign contiguous global rows, distributing any remainder to low ranks.
    int base_rows = m / processes;
    int extra_rows = m % processes;
    int local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    int first_row = rank * base_rows + min(rank, extra_rows);

    // Local row 0 and row local_rows+1 are ghost rows.
    double **u, **w;
    allocate_2d_array(local_rows + 2, n, &u);
    allocate_2d_array(local_rows + 2, n, &w);
    initialize_local_array(m, n, first_row, local_rows, u);
    initialize_local_array(m, n, first_row, local_rows, w);

    double **global_w = nullptr;
    if (rank == 0) {
        cout << "Problem size: " << m << "x" << n << "\n";
        allocate_2d_array(m, n, &global_w);
        if (max(m, n) <= VIEWABLE_SIZE) {
            double **initial;
            allocate_2d_array(m, n, &initial);
            initialize_array(m, n, initial);
            cout << "Before diffusion:\n";
            print_solution("", m, n, initial);
            delete[] initial[0]; delete[] initial;
        }
    }

    // Time only the solver. Report the slowest rank's elapsed time.
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    double final_diff;
    int its = find_steady_state(m, n, max_its, first_row, local_rows,
                                rank, processes, u, w, global_w, final_diff);
    double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (max(m, n) <= VIEWABLE_SIZE) {
            cout << "After diffusion:\n";
            print_solution("", m, n, global_w);
        }

        string status = "Converged";
        if (its > max_its) {
            status = "Not converged";
            its = max_its;
        }
        cout << status << " after " << its << " iterations with max diff: "
             << fixed << setprecision(8) << final_diff << ".\n";
        cout << "Elapsed time = " << fixed << setprecision(6) << elapsed << " sec.\n";
        print_solution(filename, m, n, global_w);
        cout << "Output file " << filename << " created.\n";

        delete[] global_w[0]; delete[] global_w;
    }

    delete[] u[0]; delete[] u;
    delete[] w[0]; delete[] w;
    MPI_Finalize();
    return 0;
}

void allocate_2d_array(int r, int c, double ***a)
{
    double *storage = new double[r * c];
    *a = new double*[r];
    for (int i = 0; i < r; i++) (*a)[i] = &storage[i * c];
}

void initialize_array(int m, int n, double **u)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) u[i][j] = 25.0;
        u[i][0] = 1000.0;
        u[i][n-1] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        u[0][j] = 0.0;
        u[m-1][j] = 0.0;
    }
}

void initialize_local_array(int m, int n, int first_row, int local_rows, double **u)
{
    for (int i = 0; i < local_rows + 2; i++) {
        int global_row = first_row + i - 1;
        for (int j = 0; j < n; j++) u[i][j] = 25.0;
        if (global_row == 0 || global_row == m-1) {
            for (int j = 0; j < n; j++) u[i][j] = 0.0;
        } else {
            u[i][0] = 1000.0;
            u[i][n-1] = 0.0;
        }
    }
}

void print_solution(const string& filename, int m, int n, double **u)
{
    ostream* outfile = &cout;
    ofstream file_stream;
    if (!filename.empty()) {
        file_stream.open(filename);
        if (!file_stream.is_open()) {
            cerr << "Can't open output file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        outfile = &file_stream;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            *outfile << setw(6) << fixed << setprecision(2) << u[i][j] << '\t';
        *outfile << '\n';
    }
}

int find_steady_state(int m, int n, int max_its, int first_row, int local_rows,
                      int rank, int processes, double **u, double **w,
                      double **global_w, double &final_diff)
{
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == processes - 1) ? MPI_PROC_NULL : rank + 1;
    double diff = 0.0;
    int its;

    for (its = 1; its <= max_its; its++) {
        // Exchange the previous iteration's edge rows into the ghost rows.
        MPI_Sendrecv(u[1], n, MPI_DOUBLE, up, 0,
                     u[local_rows + 1], n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u[local_rows], n, MPI_DOUBLE, down, 1,
                     u[0], n, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double local_diff = 0.0;
        for (int i = 1; i <= local_rows; i++) {
            int global_row = first_row + i - 1;
            if (global_row == 0 || global_row == m-1) continue;
            for (int j = 1; j < n-1; j++) {
                w[i][j] = 0.25 * (u[i-1][j] + u[i+1][j]
                                  + u[i][j-1] + u[i][j+1]);
                double d = fabs(w[i][j] - u[i][j]);
                if (d > local_diff) local_diff = d;
            }
        }

        for (int i = 1; i <= local_rows; i++) {
            int global_row = first_row + i - 1;
            if (global_row == 0 || global_row == m-1) continue;
            for (int j = 1; j < n-1; j++) u[i][j] = w[i][j];
        }

        MPI_Allreduce(&local_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (diff <= EPSILON) break;
    }

    // Every owned row, including the fixed boundary rows, belongs in the output.
    vector<int> counts, displacements;
    if (rank == 0) {
        counts.resize(processes);
        displacements.resize(processes);
        int base_rows = m / processes;
        int extra_rows = m % processes;
        for (int r = 0; r < processes; r++) {
            counts[r] = (base_rows + (r < extra_rows ? 1 : 0)) * n;
            displacements[r] = (r * base_rows + min(r, extra_rows)) * n;
        }
    }
    MPI_Gatherv(w[1], local_rows * n, MPI_DOUBLE,
                rank == 0 ? global_w[0] : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    final_diff = diff;
    return its;
}
