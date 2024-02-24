# Makefile to compile Serial, OpenMP, and CUDA implementations
# Load modules before making project
# module use /apps/modules/all
# module load GCC/11.3.0 CUDA/11.7.0 CMake/3.24.3-GCCcore-11.3.0
# module load CUDA/12.0.0

# Target rules
all: serial openmp cuda

# Compile Serial 
serial: Serial_matmult.c utils/mmio.c
	gcc -o Serial_matmult.o Serial_matmult.c utils/mmio.c

# Compile OpenMP
openmp: OpenMP_matmult.c utils/mmio.c
	gcc -fopenmp -O4 -o OpenMP_matmult OpenMP_matmult.c utils/mmio.c

# Compile CUDA
cuda: CUDA_matmult.cu utils/mmio.cu
	nvcc -g -o CUDA_matmult CUDA_matmult.cu utils/mmio.cu

clean:
	rm -f Serial_matmult.o OpenMP_matmult CUDA_matmult
