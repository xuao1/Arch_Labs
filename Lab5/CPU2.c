#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

int N = (1 << 8);
const int BLOCK_SIZE = 8;

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

/*
void transpose(float *B) {
    for(int i = 0; i < N; i++) {
        for(int j = i + 1; j < N; j++) {
            float temp = B[j * N + i];
            B[j * N + i] = B[i * N + j];
            B[i * N + j] = temp;
        }
    }
}
*/

void gemm_avx_block(float* A, float* B, float* C) {
    // transpose(B);
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE; ++ii) {
                    for (int jj = j; jj < j + BLOCK_SIZE; jj += 8) {
                        __m256 c = _mm256_loadu_ps(&C[ii * N + jj]);
                        for (int kk = k; kk < k + BLOCK_SIZE; ++kk) {
                            // float B_elements[8] = { B[(jj)*N + kk], B[(jj + 1) * N + kk], B[(jj + 2) * N + kk], B[(jj + 3) * N + kk], B[(jj + 4) * N + kk], B[(jj + 5) * N + kk], B[(jj + 6) * N + kk], B[(jj + 7) * N + kk] };
                            // __m256 b = _mm256_set_ps(B_elements[7], B_elements[6], B_elements[5], B_elements[4], B_elements[3], B_elements[2], B_elements[1], B_elements[0]);
                            c = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[ii * N + kk]), _mm256_loadu_ps(&B[kk * N + jj]), c);
                        }
                        _mm256_storeu_ps(&C[ii * N + jj], c);
                    }
                }
            }
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
        C_avx[i] = 0;
        C_base[i] = 0;
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    gemm_avx_block(A, B, C_avx);
    end = clock();

    cpu_time_used = (1000.0 * (double)(end - start)) / CLOCKS_PER_SEC;

    printf("CPU time: %f ms.\n", cpu_time_used);

    /*
    // transpose(B);
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
