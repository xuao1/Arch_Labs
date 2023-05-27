#include <cuda.h>
#include <stdio.h>

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

__global__ void matrixMulOnGPU(float* m_a, float* m_b, float* m_r, unsigned int m, unsigned int n, unsigned int k)
{
	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= m * k)
		return;

	int row = threadId / k;
	int col = threadId % k;

	m_r[threadId] = 0;
	for (size_t i = 0; i < n; ++i)
	{
		m_r[threadId] += m_a[row * n + i] * m_b[i * k + col];
	}
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
    
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    matrixMulOnGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, N, N, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    gemm_baseline(h_a, h_b, v_c);

    int flag = 1;
    for(int i = 0; i < N * N; i++) {
        printf("%d %d\n", h_c[i], v_c[i]);
        if(abs(h_c[i] - v_c[i]) > 0.00001){
            flag = 0;
            // break;
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

    return 0;
}
