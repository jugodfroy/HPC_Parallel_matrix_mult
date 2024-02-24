
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils/mmio.h"
#include <dirent.h>
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

// CUDA kernel for CSR vector/matrix dense multiplication
__global__ void multiply_csr_kernel(double *Y, const int *IRP, const int *JA, const double *AS, const double *X, int N, int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N)
    {
        for (int col = 0; col < k; col++)
        {
            double sum = 0.0;
            for (int j = IRP[row]; j < IRP[row + 1]; j++)
            {
                sum += AS[j] * X[JA[j] * k + col];
            }
            Y[row * k + col] = sum;
        }
    }
}

int main()
{
    DIR *d;
    struct dirent *dir;
    d = opendir("./matrices");
    if (!d)
    {
        perror("Failed to open directory");
        return EXIT_FAILURE;
    }

    while ((dir = readdir(d)) != NULL)
    {
        char *extension = strstr(dir->d_name, ".mtx");
        if (dir->d_type == DT_REG && extension != NULL && strcmp(extension, ".mtx") ==

                                                              0)
        {
            char filepath[1024];
            snprintf(filepath, sizeof(filepath), "./matrices/%s", dir->d_name);
            printf("Processing %s\n", filepath);

            // Flush
            fflush(stdout);

            // Open the file and proceed as before to read and process the matrix
            FILE *f = fopen(filepath, "r");
            if (!f)
            {
                perror("Error opening file");
                continue; // Skip to the next file if this one cannot be opened
            }

            int M, N, nz;
            int *IRP, *JA;
            double *AS;

            // Reading and converting the matrix to CSR format
            read_mtx_and_convert_to_csr(f, &M, &N, &nz, &IRP, &JA, &AS);
            fclose(f);

            // Values of k to test
            int k_values[] = {1, 2, 3, 6};
            int num_k_values = sizeof(k_values) / sizeof(k_values[0]);

            // Block sizes to test
            int blockSizes[] = {8, 16, 64, 128, 256};
            int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

            for (int k_index = 0; k_index < num_k_values; k_index++)
            {
                int k = k_values[k_index];

                // Initialization of vector/matrices X on host
                double **X_host = allocate_and_initialize_matrix(N, k);

                // Flatten X_host for copying to GPU
                double *X_flat = (double *)malloc(N * k * sizeof(double));
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        X_flat[i * k + j] = X_host[i][j];
                    }
                }

                // Memory allocation on GPU and data copy
                int *d_IRP, *d_JA;
                double *d_AS, *d_X, *d_Y;
                cudaMalloc((void **)&d_IRP, (M + 1) * sizeof(int));
                cudaMalloc((void **)&d_JA, nz * sizeof(int));
                cudaMalloc((void **)&d_AS, nz * sizeof(double));
                cudaMalloc((void **)&d_X, N * k * sizeof(double));
                cudaMalloc((void **)&d_Y, M * k * sizeof(double));

                cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_JA, JA, nz * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_AS, AS, nz * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_X, X_flat, N * k * sizeof(double), cudaMemcpyHostToDevice);

                for (int i = 0; i < numBlockSizes; i++)
                {
                    int blockSize = blockSizes[i];
                    int numBlocks = (N + blockSize - 1) / blockSize;

                    // Creating events for timing
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    // Start timer
                    cudaEventRecord(start);

                    // Launch kernel
                    multiply_csr_kernel<<<numBlocks, blockSize>>>(d_Y, d_IRP, d_JA, d_AS, d_X, N, k);

                    // Stop timer
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);

                    cudaDeviceSynchronize();
                    cudaError_t error = cudaGetLastError();
                    if (error != cudaSuccess)
                    {
                        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
                        // Error handling
                    }
                    // Calculate elapsed time
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    // Copy result back to host
                    double *Y_host = (double *)malloc(M * k * sizeof(double));
                    cudaMemcpy(Y_host, d_Y, M * k * sizeof(double), cudaMemcpyDeviceToHost);

                    // Calculate GFLOPS
                    float elapsed = milliseconds / 1000.0; // Convert to seconds
                    float flops = 2.0 * nz * k / elapsed;
                    printf("BlockSize: %d, numBlock: %d, k: %d, Time: %f s, GFLOPS: %f\n", blockSize, numBlocks, k, elapsed, flops / 1e9);

                    free(Y_host);
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);
                }

                // Cleanup
                cudaFree(d_X);
                cudaFree(d_Y);
                free(X_flat);
                for (int i = 0; i < N; i++)
                    free(X_host[i]);

                free(X_host);
            }

            // Final cleanup

            free(IRP);
            free(JA);
            free(AS);
        }
    }

    closedir(d);
    return 0;
}
