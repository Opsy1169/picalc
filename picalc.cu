#include <cstdlib>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
// чтобы VS не ругался на __syncthreads();
//доп. инфа здесь https://devtalk.nvidia.com/default/topic/1009723/__syncthreads-and-atomicadd-are-undefined-in-visual-studio-2015/ 
#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <device_functions.h> 
#include "device_launch_parameters.h"



#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}       

const long N = 33554432; // Points count 


__global__ void calc_PI_gpu(float *x, float *y, int *totalCount) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread id

	int threadCount = gridDim.x * blockDim.x; // Threads count to be used as a step in loop 

	int countPointsInCircle = 0;
	for (int i = idx; i < N; i += threadCount) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPointsInCircle++;
		}
	}
	atomicAdd(totalCount, countPointsInCircle); //each thread sum amount of points in circle into variable 

}


float calc_PI_CPU(float *x, float *y) {
	int countPointsInCircle = 0;
	for (int i = 0; i < N; i++) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPointsInCircle++;
		}
	}
	return float(countPointsInCircle) * 4 / N;
}



int main()
{
	float *host_X, *host_Y, *gpu_X, *gpu_Y;

	host_X = (float *)calloc(N, sizeof(float));
	host_Y = (float *)calloc(N, sizeof(float));

	CUDA_CHECK_ERROR(cudaMalloc((void **)&gpu_X, N * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void **)&gpu_Y, N * sizeof(float)));

	curandGenerator_t curandGenerator; 
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT); 
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 4321ULL);
	//generate two sequences
	curandGenerateUniform(curandGenerator, gpu_X, N); 
	curandGenerateUniform(curandGenerator, gpu_Y, N);
	curandDestroyGenerator(curandGenerator);

	CUDA_CHECK_ERROR(cudaMemcpy(host_X, gpu_X, N * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaMemcpy(host_Y, gpu_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
	

	clock_t  start_time = clock();
	float cpu_result = calc_PI_CPU(host_X, host_Y);
	clock_t  end_time = clock();
	std::cout.precision(15);
	std::cout << "CPU time = " << (double)((end_time - start_time) * 1000 / CLOCKS_PER_SEC) << " msec" << std::endl;
	std::cout << "result: " << cpu_result << std::endl;

	float gpuTime = 0;

	cudaEvent_t start;
	cudaEvent_t stop;

	int blockDim = 512;
	dim3 threads(blockDim, 1);
	dim3 grid(N / (128 * blockDim), 1);
	int *total_gpu_count;
	int *host_total_gpu_count = (int *)calloc(1, sizeof(int));

	CUDA_CHECK_ERROR(cudaMalloc((void **)&total_gpu_count, sizeof(int)));

	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));

	cudaEventRecord(start, 0);
	calc_PI_gpu << <grid, threads >> > (gpu_X, gpu_Y, total_gpu_count);

	CUDA_CHECK_ERROR(cudaMemcpy(host_total_gpu_count, total_gpu_count, sizeof(int), cudaMemcpyDeviceToHost));

	int gpu_points_count = *host_total_gpu_count;
	float gpu_result = (float)gpu_points_count * 4 / N;

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);

	std::cout << "GPU time = " << gpuTime << " мсек" << std::endl;
	std::cout << "result: " << gpu_result << std::endl;

	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));

	CUDA_CHECK_ERROR(cudaFree(gpu_X));
	CUDA_CHECK_ERROR(cudaFree(gpu_Y));
	CUDA_CHECK_ERROR(cudaFree(total_gpu_count));

	delete host_X;
	delete host_Y;
	delete host_total_gpu_count;

	system("pause");
	return 0;
}
