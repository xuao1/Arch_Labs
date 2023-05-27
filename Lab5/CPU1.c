#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

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

void gemm_avx(float *A, float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j+=8) {
            __m256 c = _mm256_setzero_ps();
            for (int k = 0; k < N; k++) {
                c = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[i*N+k]), _mm256_loadu_ps(&B[k*N+j]), c);
            }
            _mm256_storeu_ps(&C[i*N+j], c);
        }
    }
}

int main() {
	int n;
	scanf("%d", &n);
    N = 1 << n;
    
    srand(time(NULL));

    float *A = (float*) malloc(N * N * sizeof(float));
    float *B = (float*) malloc(N * N * sizeof(float));
    float *C_avx = (float*) malloc(N * N * sizeof(float));
    float *C_base = (float*) malloc(N * N * sizeof(float));

    for(int i = 0; i < N*N; i++) {
        A[i] = (float)rand()/(float)(RAND_MAX);
        B[i] = (float)rand()/(float)(RAND_MAX);
        C_avx[i] = 0.0;
        C_base[i] = 0.0;
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    gemm_avx(A, B, C_avx);
    end = clock();

    cpu_time_used = (1000.0 * (double)(end - start)) / CLOCKS_PER_SEC;

    printf("CPU time: %f ms.\n", cpu_time_used);

	/*
	// 验证
    gemm_baseline(A, B, C_base);
    
    int flag = 1;
    for(int i = 0; i < N * N; i++) {
        if(abs(C_avx[i] - C_base[i]) > 0.00001){
            flag = 0;
            break;
        }
    }

	if(flag)  printf("Results are correct.\n");
	else printf("Results are wrong.\n");
	*/
	
    free(A);
    free(B);
    free(C_avx);
    free(C_base);

    return 0;
}
