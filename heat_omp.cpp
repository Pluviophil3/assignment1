/*
 * OpenMP Jacobi solver for the 2D steady-state heat distribution problem.
 * Compile: g++ -O3 -fopenmp heat_omp.cpp -o heat_omp
 * Execute: ./heat_omp [rows] [cols] [max_its] [filename]
 * Set OMP_NUM_THREADS to choose the number of worker threads.
 */

#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

using namespace std;

const double EPSILON = 0.001;
const int VIEWABLE_SIZE = 20;

void allocate_2d_array(int r, int c, double ***a);
void initialize_array(int m, int n, double **u);
void print_solution(const string& filename, int m, int n, double **u);
int find_steady_state(int m, int n, int max_its, double **u, double **w,
                      double &final_diff, const string& filename);

int main(int argc, char *argv[])
{
    double **u;
    double **w;
    int its;
    double elapsed;
    double final_diff;

    int m = 200;
    int n = 200;
    int max_its = 100000;
    string filename;

    string prog_path(argv[0]);
    size_t pos = prog_path.find_last_of("/\\");
    string prog_name = (pos != string::npos) ? prog_path.substr(pos + 1) : prog_path;

    if (argc > 1) {
        m = stoi(argv[1]);
        n = m;
    }
    if (argc > 2) n = stoi(argv[2]);
    if (argc > 3) max_its = stoi(argv[3]);

    filename = prog_name + "-" + to_string(m) + "x" + to_string(n) + ".txt";
    if (argc > 4) filename = argv[4];
    if (argc > 5) {
        cerr << "Usage: " << argv[0] << " [rows] [cols] [max_its] [filename]\n";
        return -1;
    }

    cout << "Problem size: " << m << "x" << n << "\n";

    allocate_2d_array(m, n, &u);
    allocate_2d_array(m, n, &w);
    initialize_array(m, n, u);
    initialize_array(m, n, w);

    if (max(m, n) <= VIEWABLE_SIZE) {
        cout << "Before diffusion:\n";
        print_solution("", m, n, u);
    }

    double start = omp_get_wtime();
    its = find_steady_state(m, n, max_its, u, w, final_diff, filename);
    elapsed = omp_get_wtime() - start;

    if (max(m, n) <= VIEWABLE_SIZE) {
        cout << "After diffusion:\n";
        print_solution("", m, n, w);
    }

    string status = "Converged";
    if (its > max_its) {
        status = "Not converged";
        its = max_its;
    }

    cout << status << " after " << its << " iterations with max diff: "
         << fixed << setprecision(8) << final_diff << ".\n";
    cout << "Elapsed time = " << fixed << setprecision(6) << elapsed << " sec.\n";
    print_solution(filename, m, n, w);
    cout << "Output file " << filename << " created.\n";

    delete[] u[0]; delete[] u;
    delete[] w[0]; delete[] w;
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

void print_solution(const string& filename, int m, int n, double **u)
{
    ostream* outfile = &cout;
    ofstream file_stream;
    if (!filename.empty()) {
        file_stream.open(filename);
        if (!file_stream.is_open()) {
            cerr << "Can't open output file.\n";
            exit(-1);
        }
        outfile = &file_stream;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            *outfile << setw(6) << fixed << setprecision(2) << u[i][j] << '\t';
        *outfile << '\n';
    }
}

int find_steady_state(int m, int n, int max_its, double **u, double **w,
                      double &final_diff, const string& filename)
{
    double diff = 0.0;
    int its = 1;
    bool stop = false;
    bool converged = false;

    // Keep the team alive across iterations to avoid creating it for every loop.
    #pragma omp parallel shared(diff, its, stop, converged, u, w)
    {
        while (true) {
            #pragma omp single
            diff = 0.0;

            // Each thread owns different rows; the maximum difference is reduced
            // at the implicit barrier before the copy phase begins.
            #pragma omp for schedule(static) reduction(max:diff)
            for (int i = 1; i < m-1; i++) {
                for (int j = 1; j < n-1; j++) {
                    w[i][j] = 0.25 * (u[i-1][j] + u[i+1][j]
                                      + u[i][j-1] + u[i][j+1]);
                    double d = fabs(w[i][j] - u[i][j]);
                    if (d > diff) diff = d;
                }
            }

            // The preceding barrier ensures every read of old u is finished.
            #pragma omp for schedule(static)
            for (int i = 1; i < m-1; i++) {
                for (int j = 1; j < n-1; j++) u[i][j] = w[i][j];
            }

            #pragma omp single
            {
                converged = diff <= EPSILON;
                stop = converged || its >= max_its;
#ifdef LIVE
                if (!stop && its % 1000 == 0)
                    print_solution(filename, m, n, w);
#endif
                if (!stop) ++its;
            }
            // The single barrier publishes the shared termination decision.
            if (stop) break;
        }
    }

    final_diff = diff;
    return converged ? its : max_its + 1;
}
