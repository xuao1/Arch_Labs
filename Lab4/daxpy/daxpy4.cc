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
    for (i = 0; i < N - 3; i += 4)
    {
        Y[i] = alpha * X[i] + Y[i];
        Y[i + 1] = alpha * X[i + 1] + Y[i + 1];
        Y[i + 2] = alpha * X[i + 2] + Y[i + 2];
        Y[i + 3] = alpha * X[i + 3] + Y[i + 3];
    }
    for (; i < N; i++)
    {
        Y[i] = alpha * X[i] + Y[i];
    }
}

void daxsbxpxy_unroll(double *X, double *Y, double alpha, double beta, const int N)
{
    int i;
    for (i = 0; i < N - 3; i += 4)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
        Y[i + 1] = alpha * X[i + 1] * X[i + 1] + beta * X[i + 1] + X[i + 1] * Y[i + 1];
        Y[i + 2] = alpha * X[i + 2] * X[i + 2] + beta * X[i + 2] + X[i + 2] * Y[i + 2];
        Y[i + 3] = alpha * X[i + 3] * X[i + 3] + beta * X[i + 3] + X[i + 3] * Y[i + 3];
    }
    for (; i < N; i++)
    {
        Y[i] = alpha * X[i] * X[i] + beta * X[i] + X[i] * Y[i];
    }
}

void stencil_unroll(double *Y, double alpha, const int N)
{
    int i;
    for (i = 1; i < N - 3; i += 4)
    {
        Y[i] = alpha * Y[i - 1] + Y[i] + alpha * Y[i + 1];
        Y[i + 1] = alpha * Y[i] + Y[i + 1] + alpha * Y[i + 2];
        Y[i + 2] = alpha * Y[i + 1] + Y[i + 2] + alpha * Y[i + 3];
        Y[i + 3] = alpha * Y[i + 2] + Y[i + 3] + alpha * Y[i + 4];
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
