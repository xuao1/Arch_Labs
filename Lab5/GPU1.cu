#include <cuda.h>
#include <stdio.h>
#define BLOCK_SIZE 32

int N = (1 << 8);

void gemm_baseline(float* A, float* B, float* C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matrixMulOnGPUWithShared(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	if ((blockIdx.y * blockDim.y + threadIdx.y) * k + blockIdx.x * blockDim.x +  threadIdx.x >= m * k)
		return;

	const int begin_a = blockIdx.y * blockDim.y * n;
	const int end_a = begin_a + n - 1; 
	const int step_a = blockDim.x;

	const int begin_b = blockIdx.x * blockDim.x;
	const int step_b = blockDim.y * k;

	float result_temp = 0.0f;

	for (int index_a = begin_a, index_b = begin_b; index_a < end_a; index_a += step_a, index_b += step_b)
	{
		__shared__ float SubMat_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float SubMat_B[BLOCK_SIZE][BLOCK_SIZE];
        // 每轮循环处理A和B的BLOCK_SIZE*BLOCK_SIZE的小矩阵 

		SubMat_A[threadIdx.y][threadIdx.x] = m_a[index_a + threadIdx.y * n + threadIdx.x];
		SubMat_B[threadIdx.y][threadIdx.x] = m_b[index_b + threadIdx.y * k + threadIdx.x];
        // Share Memory是每个Block内部的，所以只需要区分Block内部的线程号

		__syncthreads(); // 确保所有线程都已经完成了数据的读取

		for (int i = 0; i < BLOCK_SIZE; ++i)
		{
			result_temp += SubMat_A[threadIdx.y][i] * SubMat_B[i][threadIdx.x];
		}

		__syncthreads(); // 确保所有线程都已经完成了计算
	}

	int begin_result = blockIdx.y * blockDim.y * k + begin_b;
	m_r[begin_result + threadIdx.y * k + threadIdx.x] = result_temp;
}

int main()
{
    int n;
    scanf("%d", &n);
    N = 1 << n;
    
    float* h_a = (float*)malloc(N * N * sizeof(float));
    float* h_b = (float*)malloc(N * N * sizeof(float));
    float* h_c = (float*)malloc(N * N * sizeof(float));
    float* v_c = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        h_a[i] = (float)rand() / (float)(RAND_MAX);
        h_b[i] = (float)rand() / (float)(RAND_MAX);
        h_c[i] = 0.0;
        v_c[i] = 0.0;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * N * sizeof(float));
    cudaMalloc((void**)&d_c, N * N * sizeof(float));
    
    cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    matrixMulOnGPUWithShared<<<gridSize, blockSize>>>(d_a, d_b, d_c, N, N, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    gemm_baseline(h_a, h_b, v_c);

    int flag = 1;
    for(int i = 0; i < N * N; i++) {
        // printf("%f %f\n", h_c[i], v_c[i]);
        if(abs(h_c[i] - v_c[i]) > 0.1){
            flag = 0;
            break;
        }
    }

	if(flag)  printf("Results are correct.\n");
	else printf("Results are wrong.\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(h_a);
    free(h_b);
    free(h_c);
    free(v_c);

    return 0;
}
