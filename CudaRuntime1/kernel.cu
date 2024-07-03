#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N 256
#define num_blocks 2
#define num_threads 128

__global__ void matrix_vector_multi_gpu(float* A_d, float* B_d, float* C_d) {
	int i, j;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	A_d[j] = 0.0F;
	for (i = 0; i < N; i++) {
		A_d[j] += B_d[j * N + i] * C_d[i];
	}
}

int main()
{
    int i, j;
	float A[N], B[N*N], C[N]; // Device
	float* A_d, * B_d, * C_d; // Host copies of A, B, C

	dim3 blocks(num_blocks, 1, 1);
	dim3 theads(num_threads, 1, 1);

    for (j = 0; j < N; j++) {
		for (i = 0; i < N; i++) {
			B[j * N + i] = ((float)j) / N;
		}
        C[j] = 1.0F;
    }

	// Memory allocation for device copies of A, B, C
	cudaMalloc((void**)&A_d, N * sizeof(float));
	cudaMalloc((void**)&B_d, N * N * sizeof(float));
	cudaMalloc((void**)&C_d, N * sizeof(float));

	// Copy inputs to device
	cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_d, C, N * sizeof(float), cudaMemcpyHostToDevice);

	matrix_vector_multi_gpu <<< blocks, theads >>> (A_d, B_d, C_d);
	cudaMemcpy(A, A_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (j = 0; j < N; j++) {
		printf("%f\n", A[j]);
    }

	// Free device memory
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	return 0;
}
