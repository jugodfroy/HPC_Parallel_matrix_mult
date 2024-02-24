#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils/mmio.h"
#include <dirent.h>
#include <signal.h>
#include <setjmp.h>
#include <string.h>

void read_mtx_and_convert_to_csr(FILE *f, int *M, int *N, int *nz, int **IRP, int **JA, double **AS)
{
    MM_typecode matcode;
    int ret_code;
    int *I, *J;
    double *val;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        exit(1);

    I = (int *)malloc(*nz * sizeof(int));
    J = (int *)malloc(*nz * sizeof(int));
    val = (double *)malloc(*nz * sizeof(double));

    *IRP = (int *)malloc((*M + 1) * sizeof(int));
    *JA = (int *)malloc(*nz * sizeof(int));
    *AS = (double *)malloc(*nz * sizeof(double));

    // Initialize IRP array to 0
    for (int i = 0; i <= *M; i++)
    {
        (*IRP)[i] = 0;
    }

    for (int i = 0; i < *nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--; // Adjust from 1-based to 0-based
        J[i]--;
        (*IRP)[I[i] + 1]++;
    }

    // Cumulative sum for IRP to determine the start positions of each row
    for (int i = 0; i < *M; i++)
    {
        (*IRP)[i + 1] += (*IRP)[i];
    }

    // Filling the JA and AS arrays
    for (int i = 0; i < *nz; i++)
    {
        int row = I[i];
        int dest = (*IRP)[row];

        (*AS)[dest] = val[i];
        (*JA)[dest] = J[i];
        (*IRP)[row]++;
    }

    // Readjustment of IRP
    int last = 0;
    for (int i = 0; i <= *M; i++)
    {
        int temp = (*IRP)[i];
        (*IRP)[i] = last;
        last = temp;
    }
}

double **allocate_and_initialize_matrix(int M, int k)
{
    double **matrix = (double **)malloc(M * sizeof(double *));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error allocating memory for the matrix.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < M; i++)
    {
        matrix[i] = (double *)malloc(k * sizeof(double));
        for (int j = 0; j < k; j++)
        {
            matrix[i][j] = rand() / (double)RAND_MAX; // Initialization with random values
        }
    }
    return matrix;
}

void multiply_csr_by_dense(double **Y, int M, int N, int *IRP, int *JA, double *AS, double **X, int k, int threads)
{
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < M; i++)
    { // For each row of the matrix A
        for (int col = 0; col < k; col++)
        { // For each column of the vector X / matrix Y
            double t = 0.0;
            for (int j = IRP[i]; j < IRP[i + 1]; j++)
            { // For each non-zero element in row i
                t += AS[j] * X[JA[j]][col];
            }
            Y[i][col] = t;
        }
    }
}

void print_matrix(double **matrix, int M, int k)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < k; j++)
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main()
{

    DIR *d;
    struct dirent *dir;
    d = opendir("./matrices"); // Open the folder containing the matrices
    if (d)
    {

        while ((dir = readdir(d)) != NULL)
        {
            char *extension = strstr(dir->d_name, ".mtx");
            // Combine file type check and extension in one condition
            if (dir->d_type == DT_REG && extension != NULL && strcmp(extension, ".mtx") == 0)
            {
                char filepath[1024];
                snprintf(filepath, sizeof(filepath), "./matrices/%s", dir->d_name); // Build the full file path

                printf("Processing %s\n", filepath);

                FILE *f = fopen(filepath, "r");
                if (!f)
                {
                    perror("Error opening file");
                    continue; // Skip to next file if this one cannot be opened
                }

                int ret_code;
                MM_typecode matcode;

                int M, N, nz; // Matrix dimensions and number of non-zero elements
                int i, *I, *J;
                double *val;
                int *IRP, *JA;
                double *AS;

                read_mtx_and_convert_to_csr(f, &M, &N, &nz, &IRP, &JA, &AS);

                // Cleanup and file close
                if (f != stdin)
                    fclose(f);

                // Initialize the random number generator with a fixed seed
                srand(42); // Fixed seed for reproducibility

                int k_values[] = {1, 2, 3, 6};
                int num_k = sizeof(k_values) / sizeof(k_values[0]);
                double **X;
                double **Y; // Result matrix Y = AX

                int nbmaxthreads = omp_get_max_threads();
                for (int threads = 1; threads <= nbmaxthreads; threads++)
                {
                    for (int k_index = 0; k_index < num_k; k_index++)
                    {
                        int k = k_values[k_index];
                        X = allocate_and_initialize_matrix(M, k);
                        Y = allocate_and_initialize_matrix(M, k);

                        double start_time = omp_get_wtime();
                        multiply_csr_by_dense(Y, M, N, IRP, JA, AS, X, k, threads);
                        double end_time = omp_get_wtime();

                        double time_taken = end_time - start_time;
                        double flops = (2.0 * nz * k) / time_taken;
                        printf("Threads: %d, k: %d, Time: %f s, GFLOPS: %f\n", threads, k, time_taken, flops / 1e9);
                    }
                }
            }
        }
    }
    return 0;
}
