# Sparse Matrix Multiplication in CUDA & OpenMP

This repository contains the implementation of sparse matrix multiplication using different high-performance computing (HPC) techniques. The project aims to compare the performance of serial, OpenMP, and CUDA implementations in multiplying a CSR (Compressed Sparse Row) formatted sparse matrix by a dense matrix.

## Directory Structure

- `matrices/` - Contains the MatrixMarket files used for testing.
- `output_example/` - Provides example outputs from the implementations.
- `submission_file/` - Submission scripts for job scheduling systems.
- `utils/` - Utility scripts and auxiliary functions.

## Implementations

- `Serial_matmult.c` - The serial implementation of sparse matrix multiplication.
- `OpenMP_matmult.c` - The OpenMP-based parallel implementation.
- `CUDA_matmult.cu` - The CUDA-based parallel implementation for GPUs.

## Compilation

A Makefile is provided for easy compilation of the project. Use the following make commands:

- `make all` - Compiles all implementations.
- `make serial` - Compiles the serial implementation.
- `make openmp` - Compiles the OpenMP implementation.
- `make cuda` - Compiles the CUDA implementation.

Ensure that the appropriate compilers (e.g., `gcc` for C/OpenMP and `nvcc` for CUDA) are available on your system.
