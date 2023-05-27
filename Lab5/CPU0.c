#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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

int main() {
    int n;
    scanf("%d", &n);
    N = 1 << n;

    srand(time(NULL));

    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / (float)(RAND_MAX);
        B[i] = (float)rand() / (float)(RAND_MAX);
        C[i] = 0.0;
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    gemm_baseline(A, B, C);
    end = clock();

    cpu_time_used = (1000.0 * (double)(end - start)) / CLOCKS_PER_SEC;

    printf("CPU time: %f ms.\n", cpu_time_used);

    free(A);
    free(B);
    free(C);

    return 0;
}
