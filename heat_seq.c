/*
 *  Sequential program that solves the 2D steady-state heat distribution problem
 *  using the Jacobi iterative method.
 *
 *  Compile:
 *  cc -O3 heat_seq.c -o heat_seq -lm
 *  cc -O3 heat_seq.c -o heat_seq_live -lm -DLIVE   (for live refresh of color map)
 *
 *  Execute:
 *  ./heat_seq [rows] [cols] [max_its] [filename]
 *
 */

#include <stdlib.h>  /* For memory allocation (malloc, free) and utilities (atoi, exit) */
#include <stdio.h>   /* For standard input/output operations (printf, fprintf, fopen, fclose) */
#include <string.h>  /* For string manipulation functions (strrchr) */
#include <math.h>    /* For mathematical functions like fabs */
#include <time.h>    /* For high-resolution timing functions (clock_gettime, struct timespec) */

#define MAX(a,b) ((a)>(b)?(a):(b))
#define EPSILON 0.001           /* Termination condition */
#define VIEWABLE_SIZE 20        /* Under this grid size, print array to console directly. */

/* Function prototypes with explicit parameters */
void allocate_2d_array(int r, int c, double ***a);
void initialize_array(int m, int n, double **u);
void print_solution(const char *filename, int m, int n, double **u);
int find_steady_state(int m, int n, int max_its, double **u, double **w, double *final_diff, const char *filename);

int main(int argc, char *argv[])
{
    double **u;                 /* Previous temperatures */
    double **w;                 /* New temperatures */
    int      its;               /* Iterations to converge */
    double   elapsed;           /* Execution time */
    double   final_diff;        /* Temperature difference at the end */
    struct timespec start, end; /* Start and end times */

    /* Default grid size and configuration */
    int m = 200;                /* Number of rows */
    int n = 200;                /* Number of cols */
    int max_its = 100000;       /* Maximum number of iterations */
    char filename[100];         /* Filename of output file */

    /* Extract base name from argv[0] (e.g., "./heat_seq" -> "heat_seq") */
    char *prog_name = strrchr(argv[0], '/');
    prog_name = (prog_name != NULL) ? prog_name + 1 : argv[0];

    /* Handle optional command-line arguments */
    if (argc > 1) {
        m = atoi(argv[1]);
        n = m;  // Default cols to match rows if only one arg is given
    }
    if (argc > 2) {
        n = atoi(argv[2]);
    }
    if (argc > 3) {
        max_its = atoi(argv[3]);
    }
    sprintf(filename, "%s-%dx%d.txt", prog_name, m, n);  // Name with program name and problem size
    if (argc > 4) {
        snprintf(filename, sizeof(filename), "%s", argv[4]);  // Override by custom filename
    }
    if (argc > 5) {
        printf("Usage: %s [rows] [cols] [max_its] [filename]\n", argv[0]);
        exit(-1);
    }

    printf("Problem size: %dx%d\n", m, n);

    allocate_2d_array(m, n, &u);
    allocate_2d_array(m, n, &w);

    initialize_array(m, n, u);
    initialize_array(m, n, w);

    if (MAX(m, n) <= VIEWABLE_SIZE) {  // For quick correctness checks
        printf("Before diffusion:\n");
        print_solution(NULL, m, n, u);
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    its = find_steady_state(m, n, max_its, u, w, &final_diff, filename);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed = (end.tv_sec - start.tv_sec) + 
              (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    if (MAX(m, n) <= VIEWABLE_SIZE) {  // For quick correctness checks
        printf("After diffusion:\n");
        print_solution(NULL, m, n, w);
    }

    char *status = "Converged";
    if (its > max_its) {
        status = "Not converged";
        its = max_its;
    }
    printf("%s after %d iterations with max diff: %8.8f.\n", status, its, final_diff);
    printf("Elapsed time = %8.6f sec.\n", elapsed);
    print_solution(filename, m, n, w);
    printf("Output file %s created.\n", filename);

    /* Clean up allocated memory */
    free(u[0]); free(u);
    free(w[0]); free(w);

    return 0;
}

/* Allocate a two-dimensional array using contiguous memory */
void allocate_2d_array(int r, int c, double ***a)
{
    double *storage = (double *) malloc(r * c * sizeof(double));
    *a = (double **) malloc(r * sizeof(double *));
    for (int i = 0; i < r; i++)
        (*a)[i] = &storage[i * c];
}

/* Initialize the temperature array */
void initialize_array(int m, int n, double **u)
{    
    /* Set initial values and boundary conditions */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            u[i][j] = 25.0;  /* Room temperature */
        }
        u[i][0] = 1000.0;    /* Heat source */
        u[i][n-1] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        u[0][j] = 0.0;
        u[m-1][j] = 0.0;
    }
}

/* Print the solution to standard output or a file */
void print_solution(const char *filename, int m, int n, double **u)
{
    int i, j;
    char sep = '\t'; 
    FILE *outfile = stdout;

    if (filename != NULL) {
        outfile = fopen(filename, "w");
        if (outfile == NULL) {
            printf("Can't open output file.");
            exit(-1);
        }
    }

    /* Print the solution array */
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            fprintf(outfile, "%6.2f%c", u[i][j], sep);
        }
        fprintf(outfile, "\n");
    }
    if (outfile != stdout)
        fclose(outfile);
}

/* Solve the steady state using Jacobi iteration */
int find_steady_state(int m, int n, int max_its, double **u, double **w, double *final_diff, const char *filename)
{
    double diff; 
    int its;
    int i, j;

    for (its = 1; its <= max_its; its++) {
        diff = 0.0;
        
        /* Compute new temperatures into w */
        for (i = 1; i < m-1; i++) {
            for (j = 1; j < n-1; j++) {
                w[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
                double d = fabs(w[i][j] - u[i][j]);
                if (d > diff) {
                    diff = d;
                }
            }
        }
        
        /* Copy w back to u for the next iteration */
        for (i = 1; i < m-1; i++) {
            for (j = 1; j < n-1; j++) {
                u[i][j] = w[i][j];
            }
        }

        /* Terminate if convergence is reached */
        if (diff <= EPSILON)
            break;

#ifdef LIVE
        if (its % 1000 == 0) {
            print_solution(filename, m, n, w);
        }
#endif
    }
    *final_diff = diff;
    return its;
}
