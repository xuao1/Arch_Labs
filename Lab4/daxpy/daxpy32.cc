#include <cstdio>
#include <random>

#include <gem5/m5ops.h>

void daxpy(double *X, double *Y, double alpha, const int N)
{
    for (int i = 0; i < N; i++)
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

void daxsbxpxy(double *X, double *Y, double alpha, double beta, const int N)
{
    for (int i = 0; i < N; i++)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
    }
}

void stencil(double *Y, double alpha, const int N)
{
    for (int i = 1; i < N-1; i++)
    {
        Y[i] = alpha * Y[i-1] + Y[i] + alpha * Y[i+1];
    }
}


void daxpy_unroll(double *X, double *Y, double alpha, const int N)
{
    int i;
    for (i = 0; i < N - 31; i += 32)
    {
        Y[i] = alpha * X[i] + Y[i];
        Y[i + 1] = alpha * X[i + 1] + Y[i + 1];
        Y[i + 2] = alpha * X[i + 2] + Y[i + 2];
        Y[i + 3] = alpha * X[i + 3] + Y[i + 3];
        Y[i + 4] = alpha * X[i + 4] + Y[i + 4];
        Y[i + 5] = alpha * X[i + 5] + Y[i + 5];
        Y[i + 6] = alpha * X[i + 6] + Y[i + 6];
        Y[i + 7] = alpha * X[i + 7] + Y[i + 7];
        Y[i + 8] = alpha * X[i + 8] + Y[i + 8];
        Y[i + 9] = alpha * X[i + 9] + Y[i + 9];
        Y[i + 10] = alpha * X[i + 10] + Y[i + 10];
        Y[i + 11] = alpha * X[i + 11] + Y[i + 11];
        Y[i + 12] = alpha * X[i + 12] + Y[i + 12];
        Y[i + 13] = alpha * X[i + 13] + Y[i + 13];
        Y[i + 14] = alpha * X[i + 14] + Y[i + 14]; 
        Y[i + 15] = alpha * X[i + 15] + Y[i + 15];
        Y[i + 16] = alpha * X[i + 16] + Y[i + 16];
        Y[i + 17] = alpha * X[i + 17] + Y[i + 17];
        Y[i + 18] = alpha * X[i + 18] + Y[i + 18];
        Y[i + 19] = alpha * X[i + 19] + Y[i + 19];
        Y[i + 20] = alpha * X[i + 20] + Y[i + 20];
        Y[i + 21] = alpha * X[i + 21] + Y[i + 21];
        Y[i + 22] = alpha * X[i + 22] + Y[i + 22];
        Y[i + 23] = alpha * X[i + 23] + Y[i + 23];
        Y[i + 24] = alpha * X[i + 24] + Y[i + 24];
        Y[i + 25] = alpha * X[i + 25] + Y[i + 25];
        Y[i + 26] = alpha * X[i + 26] + Y[i + 26];
        Y[i + 27] = alpha * X[i + 27] + Y[i + 27];
        Y[i + 28] = alpha * X[i + 28] + Y[i + 28];
        Y[i + 29] = alpha * X[i + 29] + Y[i + 29];
        Y[i + 30] = alpha * X[i + 30] + Y[i + 30];
        Y[i + 31] = alpha * X[i + 31] + Y[i + 31];
    }
    for (; i < N; i++)
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

void daxsbxpxy_unroll(double *X, double *Y, double alpha, double beta, const int N)
{
    int i;
    for (i = 0; i < N - 31; i += 32)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
        Y[i + 1] = alpha * X[i + 1] * X[i + 1] + beta * X[i + 1] + X[i + 1] * Y[i + 1];
        Y[i + 2] = alpha * X[i + 2] * X[i + 2] + beta * X[i + 2] + X[i + 2] * Y[i + 2];
        Y[i + 3] = alpha * X[i + 3] * X[i + 3] + beta * X[i + 3] + X[i + 3] * Y[i + 3];
        Y[i + 4] = alpha * X[i + 4] * X[i + 4] + beta * X[i + 4] + X[i + 4] * Y[i + 4];
        Y[i + 5] = alpha * X[i + 5] * X[i + 5] + beta * X[i + 5] + X[i + 5] * Y[i + 5];
        Y[i + 6] = alpha * X[i + 6] * X[i + 6] + beta * X[i + 6] + X[i + 6] * Y[i + 6];
        Y[i + 7] = alpha * X[i + 7] * X[i + 7] + beta * X[i + 7] + X[i + 7] * Y[i + 7];
        Y[i + 8] = alpha * X[i + 8] * X[i + 8] + beta * X[i + 8] + X[i + 8] * Y[i + 8];
        Y[i + 9] = alpha * X[i + 9] * X[i + 9] + beta * X[i + 9] + X[i + 9] * Y[i + 9];
        Y[i + 10] = alpha * X[i + 10] * X[i + 10] + beta * X[i + 10] + X[i + 10] * Y[i + 10];
        Y[i + 11] = alpha * X[i + 11] * X[i + 11] + beta * X[i + 11] + X[i + 11] * Y[i + 11];
        Y[i + 12] = alpha * X[i + 12] * X[i + 12] + beta * X[i + 12] + X[i + 12] * Y[i + 12];
        Y[i + 13] = alpha * X[i + 13] * X[i + 13] + beta * X[i + 13] + X[i + 13] * Y[i + 13];
        Y[i + 14] = alpha * X[i + 14] * X[i + 14] + beta * X[i + 14] + X[i + 14] * Y[i + 14];
        Y[i + 15] = alpha * X[i + 15] * X[i + 15] + beta * X[i + 15] + X[i + 15] * Y[i + 15];
        Y[i + 16] = alpha * X[i + 16] * X[i + 16] + beta * X[i + 16] + X[i + 16] * Y[i + 16];
        Y[i + 17] = alpha * X[i + 17] * X[i + 17] + beta * X[i + 17] + X[i + 17] * Y[i + 17];
        Y[i + 18] = alpha * X[i + 18] * X[i + 18] + beta * X[i + 18] + X[i + 18] * Y[i + 18];
        Y[i + 19] = alpha * X[i + 19] * X[i + 19] + beta * X[i + 19] + X[i + 19] * Y[i + 19];
        Y[i + 20] = alpha * X[i + 20] * X[i + 20] + beta * X[i + 20] + X[i + 20] * Y[i + 20];
        Y[i + 21] = alpha * X[i + 21] * X[i + 21] + beta * X[i + 21] + X[i + 21] * Y[i + 21];
        Y[i + 22] = alpha * X[i + 22] * X[i + 22] + beta * X[i + 22] + X[i + 22] * Y[i + 22];
        Y[i + 23] = alpha * X[i + 23] * X[i + 23] + beta * X[i + 23] + X[i + 23] * Y[i + 23];
        Y[i + 24] = alpha * X[i + 24] * X[i + 24] + beta * X[i + 24] + X[i + 24] * Y[i + 24];
        Y[i + 25] = alpha * X[i + 25] * X[i + 25] + beta * X[i + 25] + X[i + 25] * Y[i + 25];
        Y[i + 26] = alpha * X[i + 26] * X[i + 26] + beta * X[i + 26] + X[i + 26] * Y[i + 26];
        Y[i + 27] = alpha * X[i + 27] * X[i + 27] + beta * X[i + 27] + X[i + 27] * Y[i + 27];
        Y[i + 28] = alpha * X[i + 28] * X[i + 28] + beta * X[i + 28] + X[i + 28] * Y[i + 28];
        Y[i + 29] = alpha * X[i + 29] * X[i + 29] + beta * X[i + 29] + X[i + 29] * Y[i + 29];
        Y[i + 30] = alpha * X[i + 30] * X[i + 30] + beta * X[i + 30] + X[i + 30] * Y[i + 30];
        Y[i + 31] = alpha * X[i + 31] * X[i + 31] + beta * X[i + 31] + X[i + 31] * Y[i + 31];
    }
    for (; i < N; i++)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
    }
}

void stencil_unroll(double *Y, double alpha, const int N)
{
    int i;
    for (i = 1; i < N - 31; i += 32)
    {
        Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
        Y[i + 1] = alpha * Y[i] + Y[i + 1] + alpha * Y[i + 2];
        Y[i + 2] = alpha * Y[i + 1] + Y[i + 2] + alpha * Y[i + 3];
        Y[i + 3] = alpha * Y[i + 2] + Y[i + 3] + alpha * Y[i + 4];
        Y[i + 4] = alpha * Y[i + 3] + Y[i + 4] + alpha * Y[i + 5];
        Y[i + 5] = alpha * Y[i + 4] + Y[i + 5] + alpha * Y[i + 6];
        Y[i + 6] = alpha * Y[i + 5] + Y[i + 6] + alpha * Y[i + 7];
        Y[i + 7] = alpha * Y[i + 6] + Y[i + 7] + alpha * Y[i + 8];
        Y[i + 8] = alpha * Y[i + 7] + Y[i + 8] + alpha * Y[i + 9];
        Y[i + 9] = alpha * Y[i + 8] + Y[i + 9] + alpha * Y[i + 10];
        Y[i + 10] = alpha * Y[i + 9] + Y[i + 10] + alpha * Y[i + 11];
        Y[i + 11] = alpha * Y[i + 10] + Y[i + 11] + alpha * Y[i + 12];
        Y[i + 12] = alpha * Y[i + 11] + Y[i + 12] + alpha * Y[i + 13];
        Y[i + 13] = alpha * Y[i + 12] + Y[i + 13] + alpha * Y[i + 14];
        Y[i + 14] = alpha * Y[i + 13] + Y[i + 14] + alpha * Y[i + 15];
        Y[i + 15] = alpha * Y[i + 14] + Y[i + 15] + alpha * Y[i + 16];
        Y[i + 16] = alpha * Y[i + 15] + Y[i + 16] + alpha * Y[i + 17];
        Y[i + 17] = alpha * Y[i + 16] + Y[i + 17] + alpha * Y[i + 18];
        Y[i + 18] = alpha * Y[i + 17] + Y[i + 18] + alpha * Y[i + 19];
        Y[i + 19] = alpha * Y[i + 18] + Y[i + 19] + alpha * Y[i + 20];
        Y[i + 20] = alpha * Y[i + 19] + Y[i + 20] + alpha * Y[i + 21];
        Y[i + 21] = alpha * Y[i + 20] + Y[i + 21] + alpha * Y[i + 22];
        Y[i + 22] = alpha * Y[i + 21] + Y[i + 22] + alpha * Y[i + 23];
        Y[i + 23] = alpha * Y[i + 22] + Y[i + 23] + alpha * Y[i + 24];
        Y[i + 24] = alpha * Y[i + 23] + Y[i + 24] + alpha * Y[i + 25];
        Y[i + 25] = alpha * Y[i + 24] + Y[i + 25] + alpha * Y[i + 26];
        Y[i + 26] = alpha * Y[i + 25] + Y[i + 26] + alpha * Y[i + 27];
        Y[i + 27] = alpha * Y[i + 26] + Y[i + 27] + alpha * Y[i + 28];
        Y[i + 28] = alpha * Y[i + 27] + Y[i + 28] + alpha * Y[i + 29];
        Y[i + 29] = alpha * Y[i + 28] + Y[i + 29] + alpha * Y[i + 30];
        Y[i + 30] = alpha * Y[i + 29] + Y[i + 30] + alpha * Y[i + 31];
        Y[i + 31] = alpha * Y[i + 30] + Y[i + 31] + alpha * Y[i + 32];
    }
    for (; i < N - 1; i++)
    {
        Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
    }
}

int main()
{
    const int N = 10000;
    double *X = new double[N], *Y = new double[N], alpha = 0.5, beta = 0.1;

    //std::random_device rd;
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(1, 2);
    for (int i = 0; i < N; ++i)
    {
        X[i] = dis(gen);
        Y[i] = dis(gen);
    }

    m5_dump_reset_stats(0, 0);
    daxpy(X, Y, alpha, N);
    m5_dump_reset_stats(0, 0);
    daxpy_unroll(X, Y, alpha, N);
    m5_dump_reset_stats(0, 0);
    daxsbxpxy(X, Y, alpha, beta, N);
    m5_dump_reset_stats(0, 0);
    daxsbxpxy_unroll(X, Y, alpha, beta, N);
    m5_dump_reset_stats(0, 0);
    stencil(Y, alpha, N);
    m5_dump_reset_stats(0, 0);
    stencil_unroll(Y, alpha, N);
    m5_dump_reset_stats(0, 0);

    double sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += Y[i];
    }
    printf("%lf\n", sum);
    return 0;
}
